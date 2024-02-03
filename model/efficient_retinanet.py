from torchvision.models.efficientnet import *                
from torchvision.models.detection.retinanet import *
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.backbone_utils import BackboneWithFPN, ExtraFPNBlock, IntermediateLayerGetter
from torch import nn, rand
import torch
from typing import Optional, Any, List, Callable

import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops, misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.utils import _log_api_usage_once
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def _default_anchorgen():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [64, 128, 128, 256, 256, 256, 512, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator

class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        backbone,
        num_classes,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
        **kwargs,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None instead of {type(anchor_generator)}"
            )

        if anchor_generator is None:
            anchor_generator = _default_anchorgen()
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)









def _return_out_channels(model):
    layers = IntermediateLayerGetter(model, 
                                     {f"{k}": str(v) for v, k in enumerate(list(range(8,-1,-1)))})
    layers_output = layers(torch.rand((1,3,50,50)))
    return {f"{idx}": tensor.shape[1] for idx, tensor in enumerate(layers_output.values())}


def _efficientnet_fpn_extractor(
    backbone,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,8], got {trainable_layers}")
    layers_to_train = list(range(8,-1,-1))[:trainable_layers]

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(str(layer)) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [3,4,5,6,7,8]
    if min(returned_layers) <= 0 or max(returned_layers) >= 9:
        raise ValueError(f"Each returned layer should be in the range [1,8]. Got {returned_layers}")
    return_layers = {f"{k}": str(v) for v, k in enumerate(returned_layers)}
    all_channels = _return_out_channels(backbone)
    in_channels_list = [all_channels[key] for key in return_layers.keys()]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )





def get_efficient_retinanet(num_classes:Optional[int], 
                            efficientnet_model_name: Optional[str] = "efficientnet-b4", 
                            trainable_backbone_layers: Optional[int] = 0,
                            **kwargs: Any,
                            ):
    efficientnet_versions = \
        {'efficientnet-b0':(efficientnet_b0,EfficientNet_B0_Weights.IMAGENET1K_V1),
        'efficientnet-b1':(efficientnet_b1,EfficientNet_B1_Weights.IMAGENET1K_V1),
        'efficientnet-b2':(efficientnet_b2,EfficientNet_B2_Weights.IMAGENET1K_V1),
        'efficientnet-b3':(efficientnet_b3,EfficientNet_B3_Weights.IMAGENET1K_V1),
        'efficientnet-b4':(efficientnet_b4,EfficientNet_B4_Weights.IMAGENET1K_V1),
        'efficientnet-b5':(efficientnet_b5,EfficientNet_B5_Weights.IMAGENET1K_V1),
        'efficientnet-b6':(efficientnet_b6,EfficientNet_B6_Weights.IMAGENET1K_V1),
        'efficientnet-b7':(efficientnet_b7,EfficientNet_B7_Weights.IMAGENET1K_V1),
        }
    if efficientnet_model_name in efficientnet_versions.keys():
        efficientnet, backbone_weight = efficientnet_versions[efficientnet_model_name]
    else:
        raise TypeError(f"\"{efficientnet_model_name}\" is not valid. \
                        The valid model names are {list(efficientnet_versions.keys())} ")
        weights_backbone = EfficientNet_B0_Weights.verify(weights_backbone)

    if num_classes is None:
        num_classes = 91

    is_trained = backbone_weight is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 8, 5)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = efficientnet(weights=backbone_weight, progress=True, norm_layer=norm_layer).features
    backbone = _efficientnet_fpn_extractor(
                            backbone=backbone, 
                            trainable_layers=trainable_backbone_layers,
                            returned_layers=[3,4,5,6,7,8], 
                            extra_blocks=LastLevelP6P7(256,256))
    model = RetinaNet(backbone, num_classes, **kwargs)
    return model
    