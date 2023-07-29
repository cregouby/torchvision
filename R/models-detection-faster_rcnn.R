# from torchvision$ops import MultiScaleRoIAlign
#
# from ...ops import misc as misc_nn_ops
# from ...transforms._presets import ObjectDetection
# from .._api import register_model, Weights, WeightsEnum
# from .._meta import _COCO_CATEGORIES
# from .._utils import _ovewrite_value_param, handle_legacy_interface
# from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
# from ..resnet import resnet50, ResNet50_Weights
# from ._utils import overwrite_eps
# from .anchor_utils import AnchorGenerator
# from .backbone_utils import mobilenet_extractor, resnet_fpn_extractor, validate_trainable_layers
# from .generalized_rcnn import GeneralizedRCNN
# from .roi_heads import RoIHeads
# from .rpn import RegionProposalNetwork, RPNHead
# from .transform import GeneralizedRCNNTransform


all__ <- c(
  "FasterRCNN",
  "FasterRCNN_ResNet50_FPN_Weights",
  "FasterRCNN_ResNet50_FPN_V2_Weights",
  "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
  "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
  "fasterrcnn_resnet50_fpn",
  "fasterrcnn_resnet50_fpn_v2",
  "fasterrcnn_mobilenet_v3_large_fpn",
  "fasterrcnn_mobilenet_v3_large_320_fpn",
)


default_anchorgen = function() {
  anchor_sizes <- c(32, 64, 128, 256, 512)
  aspect_ratios <- rep(c(0.5, 1.0, 2.0), length(anchor_sizes))
  return(AnchorGenerator(anchor_sizes, aspect_ratios))
}

  #' Implements Faster R-CNN.
  #'
  #' The input to the model is a list of tensors, each of shape c(C, H, W), one for each
  #' image, and should be in 0-1 range. Different images can have different sizes.
  #'
  #' The behavior of the model changes depending on if it is in training or evaluation mode.
  #'
  #' During training, the model expects both the input tensors and targets (list of dictionary),
  #' containing:
  #'     - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
  #'       ``0 <= x1 < x2  <= W`` and ``0  <= y1 < y2  <= H``.
  #'     - labels (Int64Tensor[N]): the class label for each ground-truth box
  #'
  #' The model returns a Dict[Tensor] during training, containing the classification and regression
  #' losses for both the RPN and the R-CNN.
  #'
  #' During inference, the model requires only the input tensors, and returns the post-processed
  #' predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
  #' follows:
  #'     - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
  #'       ``0  <= x1 < x2  <= W`` and ``0  <= y1 < y2  <= H``.
  #'     - labels (Int64Tensor[N]): the predicted labels for each image
  #'     - scores (Tensor[N]): the scores or each prediction
  #'
  #' Args:
  #'     backbone (torch::nn_module): the network used to compute the features for the model.
  #'         It should contain an out_channels attribute, which indicates the number of output
  #'         channels that each feature map has (and it should be the same for all feature maps).
  #'         The backbone should return(a single Tensor or and OrderedDict[Tensor].)
  #'     num_classes (int): number of output classes of the model (including the background).
  #'         If box_predictor is specified, num_classes should be NULL.
  #'     min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
  #'     max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
  #'     image_mean (Tuplec(float, float, float)): mean values used for input normalization.
  #'         They are generally the mean values of the dataset on which the backbone has been trained
  #'         on
  #'     image_std (Tuplec(float, float, float)): std values used for input normalization.
  #'         They are generally the std values of the dataset on which the backbone has been trained on
  #'     rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
  #'         maps.
  #'     rpn_head (torch::nn_module): module that computes the objectness and regression deltas from the RPN
  #'     rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
  #'     rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
  #'     rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
  #'     rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
  #'     rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
  #'     rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
  #'         considered as positive during training of the RPN.
  #'     rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
  #'         considered as negative during training of the RPN.
  #'     rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
  #'         for computing the loss
  #'     rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
  #'         of the RPN
  #'     rpn_score_thresh (float): during inference, only return(proposals with a classification score)
  #'         greater than rpn_score_thresh
  #'     box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
  #'         the locations indicated by the bounding boxes
  #'     box_head (torch::nn_module): module that takes the cropped feature maps as input
  #'     box_predictor (torch::nn_module): module that takes the output of box_head and returns the
  #'         classification logits and box regression deltas.
  #'     box_score_thresh (float): during inference, only return(proposals with a classification score)
  #'         greater than box_score_thresh
  #'     box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
  #'     box_detections_per_img (int): maximum number of detections per image, for all classes.
  #'     box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
  #'         considered as positive during training of the classification head
  #'     box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
  #'         considered as negative during training of the classification head
  #'     box_batch_size_per_image (int): number of proposals that are sampled during training of the
  #'         classification head
  #'     box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
  #'         of the classification head
  #'     bbox_reg_weights (Tuplec(float, float, float, float)): weights for the encoding/decoding of the
  #'         bounding boxes
  #'
  #' Example::
  #'
  #'     >>> import torch
  #'     >>> import torchvision
  #'     >>> from torchvision$models$detection import FasterRCNN
  #'     >>> from torchvision$models$detection$rpn import AnchorGenerator
  #'     >>> # load a pre-trained model for classification and return
  #'     >>> # only the features
  #'     >>> backbone <- torchvision$models$mobilenet_v2(weights = MobileNet_V2_Weights$DEFAULT)$features
  #'     >>> # FasterRCNN needs to know the number of
  #'     >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
  #'     >>> # so we need to add it here
  #'     >>> backbone$out_channels <- 1280
  #'     >>>
  #'     >>> # let's make the RPN generate 5 x 3 anchors per spatial
  #'     >>> # location, with 5 different sizes and 3 different aspect
  #'     >>> # ratios. We have a Tuple[Tuple[int]] because each feature
  #'     >>> # map could potentially have different sizes and
  #'     >>> # aspect ratios
  #'     >>> anchor_generator <- AnchorGenerator(sizes = ((32, 64, 128, 256, 512),),
  #'     >>>                                    aspect_ratios = ((0.5, 1.0, 2.0),))
  #'     >>>
  #'     >>> # let's define what are the feature maps that we will
  #'     >>> # use to perform the region of interest cropping, as well as
  #'     >>> # the size of the crop after rescaling.
  #'     >>> # if your backbone returns a Tensor, featmap_names is expected to
  #'     >>> # be $0. More generally, the backbone should return(an)
  #'     >>> # OrderedDict[Tensor], and in featmap_names you can choose which
  #'     >>> # feature maps to use.
  #'     >>> roi_pooler <- torchvision$ops$MultiScaleRoIAlign(featmap_names = $0,
  #'     >>>                                                 output_size = 7,
  #'     >>>                                                 sampling_ratio = 2)
  #'     >>>
  #'     >>> # put the pieces together inside a FasterRCNN model
  #'     >>> model <- faster_rcnn(backbone,
  #'     >>>                    num_classes = 2,
  #'     >>>                    rpn_anchor_generator = anchor_generator,
  #'     >>>                    box_roi_pool = roi_pooler)
  #'     >>> model$eval()
  #'     >>> x <- [torch::torch_rand(3, 300, 400), torch::torch_rand(3, 500, 400)]
  #'     >>> predictions <- model(x)
faster_rcnn <- torch::nn_module(
  "faster_rcnn",
  initialize = function(
    self,
    backbone,
    num_classes = NULL,
    # transform parameters
    min_size = 800,
    max_size = 1333,
    image_mean = NULL,
    image_std = NULL,
    # RPN parameters
    rpn_anchor_generator = NULL,
    rpn_head = NULL,
    rpn_pre_nms_top_n_train = 2000,
    rpn_pre_nms_top_n_test = 1000,
    rpn_post_nms_top_n_train = 2000,
    rpn_post_nms_top_n_test = 1000,
    rpn_nms_thresh = 0.7,
    rpn_fg_iou_thresh = 0.7,
    rpn_bg_iou_thresh = 0.3,
    rpn_batch_size_per_image = 256,
    rpn_positive_fraction = 0.5,
    rpn_score_thresh = 0.0,
    # Box parameters
    box_roi_pool = NULL,
    box_head = NULL,
    box_predicton = NULL,
    box_score_thresh = 0.05,
    box_nms_thresh = 0.5,
    box_detections_per_img = 100,
    box_fg_iou_thresh = 0.5,
    box_bg_iou_thresh = 0.5,
    box_batch_size_per_image = 512,
    box_positive_fraction = 0.25,
    bbox_reg_weights = NULL
    ) {

      if (is.null(backbone$out_channels)) {
        rlang::abort(glue::glue(
          "backbone should contain an attribute out_channels ",
          "specifying the number of output channels (assumed to be the ",
          "same for all the levels)")
        )
      }
      if (!inherits(rpn_anchor_generator, "AnchorGenerator")) {
        rlang::abort(glue::glue("rpn_anchor_generator should be of type AnchorGenerator or NULL instead of {list(type(rpn_anchor_generator))}"
        ))
      }
      if (!inherits(rpn_anchor_generator, "MultiScaleRoIAlign")) {
        rlang::abort(glue::glue("box_roi_pool should be of type MultiScaleRoIAlign or NULL instead of {list(type(box_roi_pool))}"
        ))
      }

      if (!is.null(num_classes)) {
        if (!is.null(box_predictor)) {
          rlang::abort(glue::glue("num_classes should be NULL when box_predictor is specified"))
        }
      } else if (is.null(box_predictor)) {
        rlang::abort(glue::glue("num_classes should not be NULL when box_predictor is not specified"))
      }

      out_channels <- backbone$out_channels

      if (is.null(rpn_anchor_generator)) {
        rpn_anchor_generator <- default_anchorgen()
      }
      if (is.null(rpn_head)) {
        rpn_head <- RPNHead(out_channels, rpn_anchor_generator$num_anchors_per_location()[0])
      }

      rpn_pre_nms_top_n <- dict(training = rpn_pre_nms_top_n_train, testing = rpn_pre_nms_top_n_test)
      rpn_post_nms_top_n <- dict(training = rpn_post_nms_top_n_train, testing = rpn_post_nms_top_n_test)

      rpn <- RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        score_thresh = rpn_score_thresh,
      )

      if (is.null(box_roi_pool)) {
        box_roi_pool <- MultiScaleRoIAlign(featmap_names = c("0", "1", "2", "3"), output_size = 7, sampling_ratio = 2)
      }

      if (is.null(box_head)) {
        resolution <- box_roi_pool$output_size[0]
        representation_size <- 1024
        box_head <- TwoMLPHead(out_channels * resolution^2, representation_size)
      }

      if (is.null(box_predictor)) {
        representation_size <- 1024
        box_predictor <- FastRCNNPredictor(representation_size, num_classes)
      }

      roi_heads <- RoIHeads(
        # Box
        box_roi_pool,
        box_head,
        box_predictor,
        box_fg_iou_thresh,
        box_bg_iou_thresh,
        box_batch_size_per_image,
        box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
      )

      if (is.null(image_mean)) {
        image_mean <- c(0.485, 0.456, 0.406)
      }
      if (is.null(image_std)) {
        image_std <- c(0.229, 0.224, 0.225)
      }
      transform <- GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    }
  )

#' Standard heads for FPN-based models
#'
#' Args:
#'     in_channels (int): number of input channels
#'     representation_size (int): size of the intermediate representation
#'
#' @noRd
TwoMLPHead <- torch::nn_module(
  "TwoMLPHead",
  initialize = function(self, in_channels, representation_size) {
    self$fc6 <- torch::nn_linear(in_channels, representation_size)
    self$fc7 <- torch::nn_linear(representation_size, representation_size)

  },
  forward = function(self, x) {
    x <- x$flatten(start_dim = 1)
    x <- torch::nnf_relu(self$fc6(x))
    x <- torch::nnf_relu(self$fc7(x))

    return(x)
  }
)

#' Args:
#'     input_size (Tuplec(int, int, int)): the input size in CHW format.
#'     conv_layers (list): feature dimensions of each Convolution layer
#'     fc_layers (list): feature dimensions of each FCN layer
#'     norm_layer (callable, optional): Module specifying the normalization layer to use. Default: NULL
#' @noRd
FastRCNNConvFCHead <- torch::nn_module(
  "FastRCNNConvFCHead",
  initialize = function(self, input_size, conv_layers, fc_layers, norm_layer = NULL) {
    in_channels <- input_size[[1]]
    in_height <- input_size[[2]]
    in_width <- input_size[[3]]

    blocks <- torch::nn_module_list()
    previous_channels <- in_channels
    for (current_channels in conv_layers) {
      blocks$append(misc_nn_ops$Conv2dNormActivation(previous_channels, current_channels, norm_layer = norm_layer))
      previous_channels <- current_channels
    }
    blocks$append(nn$Flatten())
    previous_channels <- previous_channels * in_height * in_width
    for (current_channels in fc_layers) {
      blocks$append(torch::nn_linear(previous_channels, current_channels))
      blocks$append(torch::nn_reLU(inplace = TRUE))
      previous_channels <- current_channels
    }
    for (layer in self$modules()) {
      if (isinstance(layer, torch::nn_conv2d)) {
        torch::nn_init$kaiming_normal_(layer$weight, mode = "fan_out", nonlinearity = "relu")
        if (!is.null(layer$bias)) {
          torch::nn_init$zeros_(layer$bias)
        }
      }
    }
  }
)

#' Standard classification + bounding box regression layers
#' for Fast R-CNN.
#'
#' Args:
#'     in_channels (int): number of input channels
#'     num_classes (int): number of output classes (including background)
#' @noRd
FastRCNNPredictor <- torch::nn_module(
  "FastRCNNPredictor",
  initialize = function(self, in_channels, num_classes) {

    self$cls_score <- torch::nn_linear(in_channels, num_classes)
    self$bbox_pred <- torch::nn_linear(in_channels, num_classes * 4)

  },
  forward = function(self, x) {
    if (x$dim() == 4 && x$shape[3:4] == c(1, 1)) {
      rlang::abort(glue::glue("x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {x$shape[3:4]}"))
    }
    x <- x$flatten(start_dim = 1)
    scores <- self$cls_score(x)
    bbox_deltas <- self$bbox_pred(x)

    return(scores, bbox_deltas)
  },

  # COMMON_META <- list(
  #   "categories" = COCO_CATEGORIES,
  #   "min_size" = c(1, 1),
  # )
)

FasterRCNN_ResNet50_FPN_Weights <- function(WeightsEnum) {
  COCO_V1 <- Weights(
    url = "https:%/%download$pytorch$org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    transforms = ObjectDetection,
    meta = append(COMMON_META, list(
      "num_params" = 41755286,
      "recipe" = "https:%/%github$com/pytorch/vision/tree/main/references/detection#faster-r-cnn-resnet-50-fpn",
      "_metrics" = list(
        "COCO-val2017" = list(
          "box_map" = 37.0,
        )
      ),
      "_ops" = 134.38,
      "_file_size" = 159.743,
      "_docs" = "These weights were produced by following a similar training recipe as on the paper.",
    )),
  )
  DEFAULT <- COCO_V1
}


FasterRCNN_ResNet50_FPN_V2_Weights <-function(WeightsEnum) {
  COCO_V1 <- Weights(
    url = "https:%/%download$pytorch$org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a$pth",
    transforms = ObjectDetection,
    meta = append(COMMON_META, list(
      "num_params" = 43712278,
      "recipe" = "https:%/%github$com/pytorch/vision/pull/5763",
      "_metrics" = list(
        "COCO-val2017" = list(
          "box_map" = 46.7,
        )
      ),
      "_ops" = 280.371,
      "_file_size" = 167.104,
      "_docs" = "These weights were produced using an enhanced training recipe to boost the model accuracy.",
    )),
  )
  DEFAULT <- COCO_V1
}

FasterRCNN_MobileNet_V3_Large_FPN_Weights <- function(WeightsEnum) {
  COCO_V1 <- Weights(
    url = "https:%/%download$pytorch$org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
    transforms = ObjectDetection,
    meta = append(COMMON_META, list(
      "num_params" = 19386354,
      "recipe" = "https:%/%github$com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-fpn",
      "_metrics" = list(
        "COCO-val2017" = list(
          "box_map" = 32.8,
        )
      ),
      "_ops" = 4.494,
      "_file_size" = 74.239,
      "_docs" = "These weights were produced by following a similar training recipe as on the paper.",
    )),
  )
  DEFAULT <- COCO_V1
}


FasterRCNN_MobileNet_V3_Large_320_FPN_Weights <- function(WeightsEnum) {
  COCO_V1 <- Weights(
    url = "https:%/%download$pytorch$org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    transforms = ObjectDetection,
    meta = append(COMMON_META, list(
      "num_params" = 19386354,
      "recipe" = "https:%/%github$com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-320-fpn",
      "_metrics" = list(
        "COCO-val2017" = list(
          "box_map" = 22.8,
        )
      ),
      "_ops" = 0.719,
      "_file_size" = 74.239,
      "_docs" = "These weights were produced by following a similar training recipe as on the paper.",
    )),
  )
  DEFAULT <- COCO_V1
}

#' @handle_legacy_interface (
#'   weights = ("pretrained", FasterRCNN_ResNet50_FPN_Weights$COCO_V1),
#'   weights_backbone = ("pretrained_backbone", ResNet50_Weights$IMAGENET1K_V1)
#' )


#' Faster R-CNN model with a ResNet-50-FPN backbone from the `Faster R-CNN: Towards Real-Time Object
#' Detection with Region Proposal Networks <https:%/%arxiv$org/abs/1506.01497>`__
#' paper.
#'
#' .. betastatus:: detection module
#'
#' The input to the model is expected to be a list of tensors, each of shape ``c(C, H, W)``, one for each
#' image, and should be in ``0-1`` range. Different images can have different sizes.
#'
#' The behavior of the model changes depending on if it is in training or evaluation mode.
#'
#' During training, the model expects both the input tensors and a targets (list of dictionary),
#' containing:
#'
#'     - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
#'       ``0  <= x1 < x2  <= W`` and ``0  <= y1 < y2  <= H``.
#'     - labels (``Int64Tensor[N]``): the class label for each ground-truth box
#'
#' The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
#' losses for both the RPN and the R-CNN.
#'
#' During inference, the model requires only the input tensors, and returns the post-processed
#' predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
#' follows, where ``N`` is the number of detections:
#'
#'     - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
#'       ``0  <= x1 < x2  <= W`` and ``0  <= y1 < y2  <= H``.
#'     - labels (``Int64Tensor[N]``): the predicted labels for each detection
#'     - scores (``Tensor[N]``): the scores of each detection
#'
#' For more details on the output, you may refer to :ref:`instance_seg_output`.
#'
#' Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
#'
#' Example::
#'
#'     >>> model <- torchvision$models$detection$fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights$DEFAULT)
#'     >>> # For training
#'     >>> images, boxes <- torch::torch_rand(4, 3, 600, 1200), torch::torch_rand(4, 11, 4)
#'     >>> boxes[:, :, 2:4] <- boxes[:, :, 0:2] + boxes[:, :, 2:4]
#'     >>> labels <- torch::torch_randint(1, 91, (4, 11))
#'     >>> images <- list(image for image in images)
#'     >>> targets <- []
#'     >>> for i in range(len(images)) {
#'     >>>     d <- list()
#'     >>>     d$boxes <- boxes[i]
#'     >>>     d$labels <- labels[i]
#'     >>>     targets$append(d)
#'     >>> output <- model(images, targets)
#'     >>> # For inference
#'     >>> model$eval()
#'     >>> x <- [torch::torch_rand(3, 300, 400), torch::torch_rand(3, 500, 400)]
#'     >>> predictions <- model(x)
#'     >>>
#'     >>> # optionally, if (you want to export the model to ONNX) {
#'     >>> torch::torch_onnx$export(model, x, "faster_rcnn$onnx", opset_version <- 11)
#'
#' Args:
#'     weights (:class:`~torchvision$models$detection$FasterRCNN_ResNet50_FPN_Weights`, optional): The
#'         pretrained weights to use. See
#'         :class:`~torchvision$models$detection$FasterRCNN_ResNet50_FPN_Weights` below for
#'         more details, and possible values. By default, no pre-trained
#'         weights are used.
#'     progress (bool, optional): If TRUE, displays a progress bar of the
#'         download to stderr. Default is TRUE.
#'     num_classes (int, optional): number of output classes of the model (including the background)
#'     weights_backbone (:class:`~torchvision$models$ResNet50_Weights`, optional): The
#'         pretrained weights for the backbone.
#'     trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
#'         final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
#'         trainable. If ``NULL`` is passed (the default) this value is set to 3.
#'     ^kwargs: parameters passed to the ``torchvision$models$detection$faster_rcnn$FasterRCNN``
#'         base class. Please refer to the `source code
#'         <https:%/%github$com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn$py>`_
#'         for more details about this class.
#'
#' .. autoclass:: torchvision$models$detection$FasterRCNN_ResNet50_FPN_Weights
#'     :members:
fasterrcnn_resnet50_fpn = function(
    weights = NULL,
    progress = TRUE,
    num_classes = NULL,
    weights_backbone = ResNet50_Weights$IMAGENET1K_V1,
    trainable_backbone_layers = NULL) {
  weights <- FasterRCNN_ResNet50_FPN_Weights$verify(weights)
  weights_backbone <- ResNet50_Weights$verify(weights_backbone)

  if (!is.null(weights)) {
    weights_backbone <- NULL
    num_classes <- ovewrite_value_param("num_classes", num_classes, length(weights$meta$categories))
  }
  else if (is.null(num_classes)) {
    num_classes <- 91
  }

  is_trained <- !is.null(weights) || !is.null(weights_backbone)
  trainable_backbone_layers <- validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
  if (is_trained) {
    norm_layer <- misc_nn_ops$FrozenBatchNorm2d
  } else {
    norm_layer <- torch::nn_batchNorm2d
  }

  backbone <- resnet50(weights = weights_backbone, progress = progress, norm_layer = norm_layer)
  backbone <- resnet_fpn_extractor(backbone, trainable_backbone_layers)
  model <- FasterRCNN(backbone, num_classes = num_classes)

  if (!is.null(weights)) {
    model$load_state_dict(weights$get_state_dict(progress = progress, check_hash = TRUE))
    if (weights == FasterRCNN_ResNet50_FPN_Weights$COCO_V1) {
      overwrite_eps(model, 0.0)
    }
  }

  return(model)
}



  #' @register_model()
  #' @handle_legacy_interface(
  #'   weights = ("pretrained", FasterRCNN_ResNet50_FPN_V2_Weights$COCO_V1),
  #'   weights_backbone = ("pretrained_backbone", ResNet50_Weights$IMAGENET1K_V1),
  #' )
# }

#' Constructs an improved Faster R-CNN model with a ResNet-50-FPN backbone from `Benchmarking Detection
#' Transfer Learning with Vision Transformers <https:%/%arxiv$org/abs/2111.11429>`__ paper.
#'
#' .. betastatus:: detection module
#'
#' It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
#' :func:`~torchvision$models$detection$fasterrcnn_resnet50_fpn` for more
#' details.
#'
#' Args:
#'     weights (:class:`~torchvision$models$detection$FasterRCNN_ResNet50_FPN_V2_Weights`, optional): The
#'         pretrained weights to use. See
#'         :class:`~torchvision$models$detection$FasterRCNN_ResNet50_FPN_V2_Weights` below for
#'         more details, and possible values. By default, no pre-trained
#'         weights are used.
#'     progress (bool, optional): If TRUE, displays a progress bar of the
#'         download to stderr. Default is TRUE.
#'     num_classes (int, optional): number of output classes of the model (including the background)
#'     weights_backbone (:class:`~torchvision$models$ResNet50_Weights`, optional): The
#'         pretrained weights for the backbone.
#'     trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
#'         final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
#'         trainable. If ``NULL`` is passed (the default) this value is set to 3.
#'     ^kwargs: parameters passed to the ``torchvision$models$detection$faster_rcnn$FasterRCNN``
#'         base class. Please refer to the `source code
#'         <https:%/%github$com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn$py>`_
#'         for more details about this class.
#'
#' .. autoclass:: torchvision$models$detection$FasterRCNN_ResNet50_FPN_V2_Weights
#'     :members:
fasterrcnn_resnet50_fpn_v2 <- function(weights = NULL,
                                       progress = TRUE,
                                       num_classes = NULL,
                                       weights_backbone = NULL,
                                       trainable_backbone_layers = NULL) {
  weights <- FasterRCNN_ResNet50_FPN_V2_Weights$verify(weights)
  weights_backbone <- ResNet50_Weights$verify(weights_backbone)

  if (!is.null(weights)) {
    weights_backbone <- NULL
    num_classes <-
      ovewrite_value_param("num_classes",
                           num_classes,
                           length(weights$meta$categories))
  } else if (is.null(num_classes)) {
    num_classes <- 91
  }

  is_trained <- !is.null(weights) || !is.null(weights_backbone)
  trainable_backbone_layers <-
    validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

  backbone <-
    resnet50(weights = weights_backbone, progress = progress)
  backbone <-
    resnet_fpn_extractor(backbone,
                         trainable_backbone_layers,
                         norm_layer = torch::nn_batch_norm2d)
  rpn_anchor_generator <- default_anchorgen()
  rpn_head <-
    RPNHead(
      backbone$out_channels,
      rpn_anchor_generator$num_anchors_per_location()[0],
      conv_depth = 2
    )
  box_head <- FastRCNNConvFCHead(
    c(backbone$out_channels, 7, 7),
    c(256, 256, 256, 256),
    c(1024),
    norm_layer = torch::nn_batch_norm2d
  )
  model <- FasterRCNN(
    backbone,
    num_classes = num_classes,
    rpn_anchor_generator = rpn_anchor_generator,
    rpn_head = rpn_head,
    box_head = box_head
  )

  if (!is.null(weights)) {
    model$load_state_dict(weights$get_state_dict(progress = progress, check_hash =
                                                   TRUE))

    return(model)


  }
}

fasterrcnn_mobilenet_v3_large_fpn <- function(
    weights,
    progress,
    num_classes,
    weights_backbone,
    trainable_backbone_layers) {
  if (!is.null(weights)) {
    weights_backbone <- NULL
    num_classes <- ovewrite_value_param("num_classes", num_classes, len(weights$meta$categories))
  } else if (is.null(num_classes)) {
    num_classes <- 91
  }

  is_trained <- ( !is.null(weights) || !is.null(weights_backbone))
  trainable_backbone_layers <- validate_trainable_layers(is_trained, trainable_backbone_layers, 6, 3)
  if (condition) {
    norm_layer <- misc_nn_ops$FrozenBatchNorm2d
  } else {
    norm_layer <- torch::nn_batchNorm2d
  }

  backbone <- mobilenet_v3_large(weights = weights_backbone, progress = progress, norm_layer = norm_layer)
  backbone <- mobilenet_extractor(backbone, TRUE, trainable_backbone_layers)
  anchor_sizes <- rep( c(32,64, 128,  256,  512), 3)
  aspect_ratios <- rep( c(0.5, 1.0, 2.0), length(anchor_sizes))
  model <- FasterRCNN(
    backbone, num_classes, rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
  )

  if (!is.null(weights)) {
    model$load_state_dict(weights$get_state_dict(progress = progress, check_hash = TRUE))
  }

  return(model)
}

#' @handle_legacy_interface(
#'   weights = ("pretrained", FasterRCNN_MobileNet_V3_Large_320_FPN_Weights$COCO_V1),
#'   weights_backbone = ("pretrained_backbone", MobileNet_V3_Large_Weights$IMAGENET1K_V1),
#' )
''
#' Low resolution Faster R-CNN model with a MobileNetV3-Large backbone tuned for mobile use cases.
#'
#' .. betastatus:: detection module
#'
#' It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
#' :func:`~torchvision$models$detection$fasterrcnn_resnet50_fpn` for more
#' details.
#'
#' Example::
#'
#'     >>> model <- torchvision$models$detection$fasterrcnn_mobilenet_v3_large_320_fpn(weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights$DEFAULT)
#'     >>> model$eval()
#'     >>> x <- [torch::torch_rand(3, 300, 400), torch::torch_rand(3, 500, 400)]
#'     >>> predictions <- model(x)
#'
#' Args:
#'     weights (:class:`~torchvision$models$detection$FasterRCNN_MobileNet_V3_Large_320_FPN_Weights`, optional): The
#'         pretrained weights to use. See
#'         :class:`~torchvision$models$detection$FasterRCNN_MobileNet_V3_Large_320_FPN_Weights` below for
#'         more details, and possible values. By default, no pre-trained
#'         weights are used.
#'     progress (bool, optional): If TRUE, displays a progress bar of the
#'         download to stderr. Default is TRUE.
#'     num_classes (int, optional): number of output classes of the model (including the background)
#'     weights_backbone (:class:`~torchvision$models$MobileNet_V3_Large_Weights`, optional): The
#'         pretrained weights for the backbone.
#'     trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
#'         final block. Valid values are between 0 and 6, with 6 meaning all backbone layers are
#'         trainable. If ``NULL`` is passed (the default) this value is set to 3.
#'     ^kwargs: parameters passed to the ``torchvision$models$detection$faster_rcnn$FasterRCNN``
#'         base class. Please refer to the `source code
#'         <https:%/%github$com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn$py>`_
#'         for more details about this class.
#'
#' .. autoclass:: torchvision$models$detection$FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
#'     :members:
fasterrcnn_mobilenet_v3_large_320_fpn <- function(
    weights = NULL,
    progress = TRUE,
    num_classes = NULL,
    weights_backbone = MobileNet_V3_Large_Weights$IMAGENET1K_V1,
    trainable_backbone_layers = NULL
) {
  weights <- FasterRCNN_MobileNet_V3_Large_320_FPN_Weights$verify(weights)
  weights_backbone <- MobileNet_V3_Large_Weights$verify(weights_backbone)

  defaults <- list(
    "min_size" = 320,
    "max_size" = 640,
    "rpn_pre_nms_top_n_test" = 150,
    "rpn_post_nms_top_n_test" = 150,
    "rpn_score_thresh" = 0.05
  )

  # kwargs <- list(^defaults, ^kwargs)
  return(fasterrcnn_mobilenet_v3_large_fpn(
    weights = weights,
    progress = progress,
    num_classes = num_classes,
    weights_backbone = weights_backbone,
    trainable_backbone_layers = trainable_backbone_layers)
  )
}

#' @handle_legacy_interface(
#'   weights = ("pretrained", FasterRCNN_MobileNet_V3_Large_FPN_Weights$COCO_V1),
#'   weights_backbone = ("pretrained_backbone", MobileNet_V3_Large_Weights$IMAGENET1K_V1),
#' )
# }

#' Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone.
#'
#' .. betastatus:: detection module
#'
#' It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
#' :func:`~torchvision$models$detection$fasterrcnn_resnet50_fpn` for more
#' details.
#'
#' Example::
#'
#'     >>> model <- torchvision$models$detection$fasterrcnn_mobilenet_v3_large_fpn(weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights$DEFAULT)
#'     >>> model$eval()
#'     >>> x <- [torch::torch_rand(3, 300, 400), torch::torch_rand(3, 500, 400)]
#'     >>> predictions <- model(x)
#'
#' Args:
#'     weights (:class:`~torchvision$models$detection$FasterRCNN_MobileNet_V3_Large_FPN_Weights`, optional): The
#'         pretrained weights to use. See
#'         :class:`~torchvision$models$detection$FasterRCNN_MobileNet_V3_Large_FPN_Weights` below for
#'         more details, and possible values. By default, no pre-trained
#'         weights are used.
#'     progress (bool, optional): If TRUE, displays a progress bar of the
#'         download to stderr. Default is TRUE.
#'     num_classes (int, optional): number of output classes of the model (including the background)
#'     weights_backbone (:class:`~torchvision$models$MobileNet_V3_Large_Weights`, optional): The
#'         pretrained weights for the backbone.
#'     trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
#'         final block. Valid values are between 0 and 6, with 6 meaning all backbone layers are
#'         trainable. If ``NULL`` is passed (the default) this value is set to 3.
#'     ^kwargs: parameters passed to the ``torchvision$models$detection$faster_rcnn$FasterRCNN``
#'         base class. Please refer to the `source code
#'         <https:%/%github$com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn$py>`_
#'         for more details about this class.
#'
#' .. autoclass:: torchvision$models$detection$FasterRCNN_MobileNet_V3_Large_FPN_Weights
#'     :members:
fasterrcnn_mobilenet_v3_large_fpn = function(
    weights = NULL,
    progress = TRUE,
    num_classes = NULL,
    weights_backbone = MobileNet_V3_Large_Weights$IMAGENET1K_V1,
    trainable_backbone_layers = NULL) {
  weights <- FasterRCNN_MobileNet_V3_Large_FPN_Weights$verify(weights)
  weights_backbone <- MobileNet_V3_Large_Weights$verify(weights_backbone)

  defaults <- list(
    "rpn_score_thresh" = 0.05,
  )

  # kwargs <- list(^defaults, ^kwargs)
  return(fasterrcnn_mobilenet_v3_large_fpn(
    weights = weights,
    progress = progress,
    num_classes = num_classes,
    weights_backbone = weights_backbone,
    trainable_backbone_layers = trainable_backbone_layers
  )
  )
}
