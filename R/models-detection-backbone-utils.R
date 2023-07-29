# import warnings
# from typing import Callable, Dict, List, Optional, Union
#
# from torch import nn, Tensor
# from torchvision$ops import misc as misc_nn_ops
# from torchvision$ops$feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
#
# from .. import mobilenet, resnet
# from .._api import _get_enum_from_fn, WeightsEnum
# from .._utils import handle_legacy_interface, IntermediateLayerGetter


#' Adds a FPN on top of a model.
#' Internally, it uses torchvision$models._utils$IntermediateLayerGetter to
#' extract a submodel that returns the feature maps specified in return_layers.
#' The same limitations of IntermediateLayerGetter apply here.
#' Args:
#'     backbone (torch::nn_module)
#'     return_layers (Dict c(name, new_name)): a dict containing the names
#'         of the modules for which the activations will be returned as
#'         the key of the dict, and the value of the dict is the name
#'         of the returned activation (which the user can specify).
#'     in_channels_list (List[int]): number of channels for each feature map
#'         that is returned, in the order they are present in the OrderedDict
#'     out_channels (int): number of channels in the FPN.
#'     norm_layer (callable, optional): Module specifying the normalization layer to use. Default: NULL
#' Attributes:
#'     out_channels (int): the number of channels in the FPN
backbone_with_fpn <- torch::nn_module(
  "BackboneWithFPN",
  initialize = function(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_block = NULL,
        norm_laye = NULL) {


        if (is.null(extra_blocks)) {
            extra_blocks <- LastLevelMaxPool()
        }
        self$body <- IntermediateLayerGetter(backbone, return_layer = return_layers)
        self$fpn <- FeaturePyramidNetwork(
            in_channels_lis = in_channels_list,
            out_channel = out_channels,
            extra_block = extra_blocks,
            norm_laye = norm_layer,
        )
        self$out_channels <- out_channels

    },
  forward = function(self, x) {
          x <- self$body(x)
          x <- self$fpn(x)
          return(x)
  }

)


#' @handle_legacy_interface(
#'     weight = (
#'         "pretrained",
#'         lambda kwargs: _get_enum_from_fn(resnet.__dict__[kwargs$backbone_name])$IMAGENET1K_V1,
#'     ),
#' )



#' Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.
#'
#' Examples::
#'
#'     >>> from torchvision$models$detection$backbone_utils import resnet_fpn_backbone
#'     >>> backbone <- resnet_fpn_backbone('resnet50', weight = ResNet50_Weights$DEFAULT, trainable_layer = 3)
#'     >>> # get some dummy image
#'     >>> x <- torch::torch_rand(1,3,64,64)
#'     >>> # compute the output
#'     >>> output <- backbone(x)
#'     >>> print(c((k, v$shape) for k, v in output$items()))
#'     >>> # returns
#'     >>>   [('0', torch::torch_Size([1, 256, 16, 16])),
#'     >>>    ('1', torch::torch_Size([1, 256, 8, 8])),
#'     >>>    ('2', torch::torch_Size([1, 256, 4, 4])),
#'     >>>    ('3', torch::torch_Size([1, 256, 2, 2])),
#'     >>>    ('pool', torch::torch_Size([1, 256, 1, 1]))]
#'
#' Args:
#'     backbone_name (string): resnet architecture. Possible values are 'resnet18', 'resnet34', 'resnet50',
#'          'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
#'     weights (WeightsEnum, optional): The pretrained weights for the model
#'     norm_layer (callable): it is recommended to use the default value. For details visit:
#'         (https:%/%github$com/facebookresearch/maskrcnn-benchmark/issues/267)
#'     trainable_layers (int): number of trainable (not frozen) layers starting from final block.
#'         Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
#'     returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
#'         By default, all layers are returned.
#'     extra_blocks (ExtraFPNBlock or NULL): if provided, extra operations will
#'         be performed. It is expected to take the fpn features, the original
#'         features and the names of the original features as input, and returns
#'         a new list of feature maps and their corresponding names. By
#'         default, a ``LastLevelMaxPool`` is used.
resnet_fpn_backbone <- function(
    backbone_name,
    weights,
    norm_laye = misc_nn_ops$FrozenBatchNorm2d,
    trainable_layer = 3,
    returned_layer = NULL,
    extra_block = NULL) {
    backbone <- resnet.dict[backbone_name](weight = weights, norm_laye = norm_layer)
    return(resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks))
}


resnet_fpn_extractor <- function(backbone, trainable_layers, returned_layer = NULL, extra_block = NULL, norm_laye = NULL) {

    # select layers that won't be frozen
    if (trainable_layers < 1 || trainable_layers > 6) {
        rlang::abort(glue::glue("Trainable layers should be in the range [1,6], got {trainable_layers}"))
    }
    layers_to_train <- c("layer4", "layer3", "layer2", "layer1", "conv1", "bn1")[1:trainable_layers] %>% paste0(collapse="|")

    name <- names(backbone$named_parameters())
    name <- name[!str_starts(name, layers_to_train)]
    for (nam in name) {
      backbone$nam$parameters$weight$requires_grad <- FALSE
    }

    if (is.null(extra_blocks)) {
        extra_blocks <- LastLevelMaxPool()
    }
    if (is.null(returned_layers)) {
        returned_layers <- c(1, 2, 3, 4)
    }
    if (min(returned_layers) == 0 || max(returned_layers) == 5) {
        rlang::abort(glue::glue("Each returned layer should be in the range [1,4]. Got {returned_layers}"))
    }
    return_layers <- glue::glue("layer{returned_layers}")

    in_channels_stage2 <-
    in_channels_list <- 2^(returned_layers - 1) * (backbone$inplanes %/% 8)
    out_channels <- 256
    return(backbone_with_fpn(
        backbone, return_layers, in_channels_list, out_channels, extra_block = extra_blocks, norm_laye = norm_layer
    )
  )

}
validate_trainable_layers <- function(
    is_trained,
    trainable_backbone_layers,
    max_value,
    default_value) {

    # don't freeze any layers if pretrained model or backbone is not used
    if (!is_trained) {
        if (!is.null(trainable_backbone_layers)) {
            rlang::warn(glue::glue(
                "Changing trainable_backbone_layers has no effect if ",
                "neither pretrained nor pretrained_backbone have been set to TRUE, ",
                "falling back to trainable_backbone_layer = {max_value} so that all layers are trainable"
            ))
        }
        trainable_backbone_layers <- max_value
    }
    # by default freeze first blocks
    if (is.null(trainable_backbone_layers)) {
        trainable_backbone_layers <- default_value
    }
    if (trainable_backbone_layers < 0 || trainable_backbone_layers > max_value) {
        rlang::abort(glue::glue(
            "Trainable backbone layers should be in the range [0,{max_value}], got {trainable_backbone_layers} "
        ))
    }
    return(trainable_backbone_layers)
}


#' @handle_legacy_interface(
#'     weight = (
#'         "pretrained",
#'         lambda kwargs: _get_enum_from_fn(mobilenet.__dict__[kwargs$backbone_name])$IMAGENET1K_V1,
#'     ),
#' )

mobilenet_backbone <- function(
    backbone_name,
    weights,
    fpn,
    norm_layer = misc_nn_ops$FrozenBatchNorm2d,
    trainable_layers = 2,
    returned_layers = NULL,
    extra_blocks = NULL) {
    backbone <- mobilenet.__dict__[backbone_name](weight = weights, norm_laye = norm_layer)
    return(mobilenet_extractor(backbone, fpn, trainable_layers, returned_layers, extra_blocks))
}


mobilenet_extractor <- function(
    backbone,
    fpn,
    trainable_layers,
    returned_layers = NULL,
    extra_blocks = NULL,
    norm_layer = NULL) {
  backbone <- backbone$features
  # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
  # The first and last blocks are always included because they are the C0 (conv1) and Cn.
  backbone_feat_is_cn <- paste0("backbone$`",backbone$children %>% names,"`$.is_cn")
  stage_indices <- c(1,  which(sapply(backbone_feat_is_cn, function(x) eval(parse(text = x))) %>% as.character() == "FALSE") ,length(backbone_feat_is_cn))
  num_stages <- length(stage_indices)

  # find the index of the layer from which we won't freeze
  if (trainable_layers < 0 || trainable_layers > num_stages) {
    rlang::abort(glue::glue("Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} "))
  }
  if (trainable_layers == 0 ) {
    freeze_before <- length(backbone)
  } else {
    freeze_before <- stage_indices[num_stages - trainable_layers]
  }

  for (name in 1:freeze_before) {
    backbone$name$parameters$weight$requires_grad <- FALSE
  }

  out_channels <- 256
  if (fpn) {
    if (is.null(extra_blocks)) {
        extra_blocks <- LastLevelMaxPool()
    }
    if (is.null(returned_layers)) {
        returned_layers <- c(num_stages - 1, num_stages)
    }
    if (min(returned_layers) < 0 || max(returned_layers) == num_stages) {
        rlang::abort(glue::glue("Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} "))
    }
    return_layers <- intersect(stage_indices, returned_layers)

    backbone_out_channels <- paste0("backbone$`",stage_indices[returned_layers],"`$out_channels")
    in_channels_list <- sapply(backbone_out_channels, function(x) eval(parse(text = x)))
    return(backbone_with_fpn(
        backbone, return_layers, in_channels_list, out_channels, extra_block = extra_blocks, norm_laye = norm_layer
    ))
  } else {
    m <- torch::nn_sequential(
        backbone,
        # depthwise linear combination of channels to reduce their size
        torch::nn_conv2d(tail(backbone,1)$out_channels, out_channels, 1),
    )
    m$out_channels <- out_channels  # type: ignore[assignment]
    return(m)
  }
}
