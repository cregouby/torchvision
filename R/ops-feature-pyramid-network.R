# from collections import OrderedDict
# from typing import Callable, Dict, List, Optional, Tuple
#
# import torch::torch_nn$functional as F
# from torch import nn, Tensor
#
# from ..ops$misc import Conv2dNormActivation
# from ..utils import _log_api_usage_once
#

#' Base class for the extra block in the FPN.
#'
#' Args:
#'     results (List[Tensor]): the result of the FPN
#'     x (List[Tensor]): the original feature maps
#'     names (List[str]): the names for each one of the
#'         original feature maps
#'
#' Returns:
#'     results (List[Tensor]): the extended set of results
#'         of the FPN
#'     names (List[str]): the extended set of names for the results
#'
ExtraFPNBlock <- torch::nn_module(
  "ExtraFPNBlock",
  forward = function(self, results, x, names ) {
          pass
    }
)

#' Module that adds a FPN from on top of a set of feature maps. This is based on
#' `"Feature Pyramid Network for Object Detection" <https:%/%arxiv$org/abs/1612.03144>`_.
#'
#' The feature maps are currently supposed to be in increasing depth
#' order.
#'
#' The input to the model is expected to be an OrderedDict[Tensor], containing
#' the feature maps on top of which the FPN will be added.
#'
#' Args:
#'     in_channels_list (list[int]): number of channels for each feature map that
#'         is passed to the module
#'     out_channels (int): number of channels of the FPN representation
#'     extra_blocks (ExtraFPNBlock or NULL): if provided, extra operations will
#'         be performed. It is expected to take the fpn features, the original
#'         features and the names of the original features as input, and returns
#'         a new list of feature maps and their corresponding names
#'     norm_layer (callable, optional): Module specifying the normalization layer to use. Default: NULL
#'
#' Examples::
#'
#'     >>> m <- torchvision$ops$FeaturePyramidNetwork([10, 20, 30], 5)
#'     >>> # get some dummy data
#'     >>> x <- OrderedDict()
#'     >>> x$feat0 <- torch::torch_rand(1, 10, 64, 64)
#'     >>> x$feat2 <- torch::torch_rand(1, 20, 16, 16)
#'     >>> x$feat3 <- torch::torch_rand(1, 30, 8, 8)
#'     >>> # compute the FPN on top of x
#'     >>> output <- m(x)
#'     >>> print(c((k, v$shape) for k, v in output$items()))
#'     >>> # returns
#'     >>>   list('feat0', torch::torch_Size([1, 5, 64, 64])),
#'     >>>    ('feat2', torch::torch_Size([1, 5, 16, 16])),
#'     >>>    ('feat3', torch::torch_Size([1, 5, 8, 8]))
feature_pyramid_network <- torch::nn_module(
  "feature_pyramid_network",
  # _version <- 2,
  initialize = function(
    self,
    in_channels_list,
    out_channels,
    extra_blocks = NULL,
    norm_layer = NULL) {

    # _log_api_usage_once(self)
    self$inner_blocks <- torch::nn_moduleList()
    self$layer_blocks <- torch::nn_moduleList()
    for (in_channels in in_channels_list) {
      if (in_channels == 0) {
        rlang::abort(glue::glue("in_channels=0 is currently not supported"))
      }
      inner_block_module <- Conv2dNormActivation(
        in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=NULL
      )
      layer_block_module <- Conv2dNormActivation(
        out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=NULL
      )
      self$inner_blocks$append(inner_block_module)
      self$layer_blocks$append(layer_block_module)
    }

    # initialize parameters now to avoid modifying the initialization of top_blocks
    for (m in self$modules()) {
      if (inherits(m, "nn_conv2d")){
        torch::nn_init$kaiming_uniform_(m$weight, a=1)
        if (!m$is.null(bias)) {
          torch::nn_init$constant_(m$bias, 0)
        }
      }
    }

    if (!is.null(extra_blocks)) {
      if (!inherits(extra_blocks, ExtraFPNBlock)) {
        rlang::abort(glue::glue("extra_blocks should be of type ExtraFPNBlock not {str(extra_blocks)}"))
      }
    }

    self$extra_blocks <- extra_blocks
  },

  load_from_state_dict = function(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs    ) {
    version <- local_metadata$get("version", NULL)

    if (is.null(version) || version < 2) {
      num_blocks <- length(self$inner_blocks)
      for (block in c("inner_blocks", "layer_blocks")) {
        for (i in range(num_blocks)) {
          for (type in c("weight", "bias")) {
            old_key <- glue::glue("list(prefix}{block}.{i}.{type)")
            new_key <- glue::glue("list(prefix}{block}.{i}.0.{type)")
            if (old_key %in% state_dict) {
              state_dict[new_key] <- state_dict$pop(old_key)
            }
          }
        }
      }
    }


    load_from_state_list(
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs
      )

  },

  # This is equivalent to self$inner_blocks[idx](x),
  # but torchscript doesn't support this yet
  get_result_from_inner_blocks = function(self, x, idx) {
    num_blocks <- length(self$inner_blocks)
    if (idx < 0) {
      idx <- idx + num_blocks
    }
    out <- x
    for (i in seq_len(self$inner_blocks)) {
      if (i == idx) {
        out <- self$inner_blocks[[i]](x)
      }
    }
    return(out)
  },

# """
# This is equivalent to self$layer_blocks[idx](x),
# but torchscript doesn't support this yet
# """
  get_result_from_layer_blocks = function(self, x, idx) {
    num_blocks <- length(self$layer_blocks)
    if (idx < 0) {
      idx = idx + num_blocks
    }
    out <- x
    for (i in seq_len(self$layer_blocks)) {
      if (i == idx) {
        out <- self$layer_blocks[[i]](x)
      }
    }
    return(out)
  },

  # Computes the FPN for a set of feature maps.
  #
  # Args:
  #     x (OrderedDict[Tensor]): feature maps for each feature level.
  #
  # Returns:
  #     results (OrderedDict[Tensor]): feature maps after FPN layers.
  #         They are ordered from the highest resolution first.
  forward = function(self, x) {
    # unpack OrderedDict into two lists for easier handling
    names <- x$keys()
    x <- x$values()

    last_inner <- self$get_result_from_inner_blocks(x[-1], -1)
    results <- list()
    results$append(self$get_result_from_layer_blocks(last_inner, -1))

    for (idx in seq(length(x) - 2, 0, -1)) {
      inner_lateral <- self$get_result_from_inner_blocks(x[idx], idx)
      feat_shape <- inner_lateral$shape[N-2:N]
      inner_top_down <- torch::nnf_interpolate(last_inner, size = feat_shape, mode = "nearest")
      last_inner <- inner_lateral + inner_top_down
      results$insert(0, self$get_result_from_layer_blocks(last_inner, idx))
    }
    if (!self$is.null(extra_blocks)) {
      c(results, names) %<-% self$extra_blocks(results, x, names)
    }
    # make it back an OrderedDict
    # TODO need rework
    out <- c(names, results)

    return(out)

  }
)

# Applies a max_pool2d (not actual max_pool2d, we just subsample) on top of the last feature map
LastLevelMaxPool <- torch::nn_module(
  "ExtraFPNBlock",
  forward = function(
        self,
        x,
        y,
        names
    ) {
        names$append("pool")
        # Use max pooling to simulate stride 2 subsampling
        x$append(torch::nnf_max_pool2d(x[-1], kernel_size=1, stride=2, padding=0))
        return(x, names)
  }
)


# This module is used in RetinaNet to generate extra layers, P6 and P7.
LastLevelP6P7 <- torch::nn_module(
  "ExtraFPNBlock",
  initialize = function(self, in_channels, out_channels) {

        self$p6 <- torch::nn_conv2d(in_channels, out_channels, 3, 2, 1)
        self$p7 <- torch::nn_conv2d(out_channels, out_channels, 3, 2, 1)
        for (module in c(self$p6, self$p7)) {
            torch::nn_init$kaiming_uniform_(module$weight, a=1)
            torch::nn_init$constant_(module$bias, 0)
        }

        self$use_P5 <- in_channels == out_channels
  },
  forward = function(self, p, c, names) {
          p5 <- p[-1]
          c5 <- c[-1]
          if (self$use_P5) {
            x <- p5
          } else {
            x <- c5
          }
          p6 <- self$p6(x)
          p7 <- self$p7(torch::nnf_relu(p6))
          p$extend(c(p6, p7))
          names$extend(c("p6", "p7"))
          return(p, names)
  }
)
