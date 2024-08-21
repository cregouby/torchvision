#' from .roi_align import roi_align
#'
#'
#' # copying result_idx_in_level to a specific index in result[]
#' # is not supported by ONNX tracing yet.
#' # _onnx_merge_levels() is an implementation supported by ONNX
#' # that merges the levels to the right indices
#' @torch::torch_jit$unused
#' }
#' _onnx_merge_levels = function(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
#'     first_result <- unmerged_results[0]
#'     dtype, device <- first_result$dtype, first_result$device
#'     res <- torch::torch_zeros(
#'         (levels$size(0), first_result$size(1), first_result$size(2), first_result$size(3)), dtype = dtype, device = device
#'     )
#'     for level in range(length(unmerged_results)) {
#'         index <- torch::torch_where(levels == level)[0]$view(-1, 1, 1, 1)
#'         index <- index$expand(
#'             index$size(0),
#'             unmerged_results[level]$size(1),
#'             unmerged_results[level]$size(2),
#'             unmerged_results[level]$size(3),
#'         )
#'         res <- res$scatter(0, index, unmerged_results[level])
#'     return(res)
#'
#'
#' # TODO: (eellison) T54974082 https:%/%github$com/pytorch/pytorch/issues/26744/pytorch/issues/26744
#' }


#' Determine which FPN level each RoI in a set of RoIs should map to based
#' on the heuristic in the FPN paper.
#'
#' Args:
#'     k_min (int)
#'     k_max (int)
#'     canonical_scale (int)
#'     canonical_level (int)
#'     eps (float)
initLevelMapper <- function(
    k_min,
    k_max,
    canonical_scale = 224,
    canonical_level = 4,
    eps = 1e-6) {
  return(LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps))
}

LevelMapper <-  torch::nn_module(
  "MultiScaleRoIAlign",
  initialize = function(
    self,
    k_min,
    k_max,
    canonical_scale = 224,
    canonical_level = 4,
    eps = 1e-6
  ) {
    self$k_min <- k_min
    self$k_max <- k_max
    self$s0 <- canonical_scale
    self$lvl0 <- canonical_level
    self$eps <- eps

  },
  # Args:
  #     boxlists (list[BoxList])
  call = function(self, boxlists) {
    # Compute level ids
    s <- torch::torch_sqrt(torch::torch_cat(map(boxlists, ~box_area(.x))))

    # Eqn.(1) in FPN paper
    target_lvls <- torch::torch_floor(self$lvl0 + torch::torch_log2(s / self$s0) + torch::torch_tensor(self$eps, dtype = s$dtype))
    target_lvls <- torch::torch_clamp(target_lvls, min = self$k_min, max = self$k_max)
    return((target_lvls$to(torch::torch_int64) - self$k_min)$to(torch::torch_int64))
  }
)


convert_to_roi_format = function(boxes) {
    concat_boxes <- torch::torch_cat(boxes, dim = 0)
    device <- concat_boxes$device
    dtype <- concat_boxes$dtype
    ids <- torch::torch_cat(
        purrr::map(seq_len(boxes),
                   ~torch::torch_full_like(boxes[[.x]][, 1:2], .x, dtype = dtype, layout = torch::torch_strided, device = device)
        ),
        dim = 0
    )
    rois <- torch::torch_cat(c(ids, concat_boxes), dim = 1)
    return(rois)
}


infer_scale = function(feature, original_shape) {
    # assumption: the scale is of the form 2 ^ (-k), with k integer
    s1 <- tail(feature$shape, 2)[1]
    s2 <- tail(original_shape, 2)[1]
    return(2 ^ (round(log2(s1/s2))))
}


setup_scales = function(features, image_shapes, canonical_scale, canonical_level) {
    if (!image_shapes) {
        rlang::abort(glue::glue("image_shapes list should not be empty"))
    }

    original_input_shape <- c(max(map_dbl(image_shapes, ~.x[1])), max(map_dbl(image_shapes, ~.x[2])))

    scales <- map_dbl(features, ~infer_scale(.x, original_input_shape))
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    map_levels <- initLevelMapper(
      round(-log2(head(scales, 1))),
      round(-log2(tail(scales, 1))),
      canonical_scale = canonical_scale,
      canonical_level = canonical_level,
    )
    return(list(scales, map_levels))
}

#'
#' @torch::torch_fx$wrap
#' }
#' _filter_input = function(x: Dictc(str, Tensor), featmap_names: List[str]) -> List[Tensor]:
#'     x_filtered <- []
#'     for k, v in x$items() {
#'         if (k in featmap_names) {
#'             x_filtered$append(v)
#'     return(x_filtered)
#'
#'
#' @torch::torch_fx$wrap
#' }
#'
#'

#' Args:
#'     x_filtered (List[Tensor]): List of input tensors.
#'     boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
#'         (x1, y1, x2, y2) format and in the image reference size, not the feature map
#'         reference. The coordinate must satisfy ``0 < = x1 < x2`` and ``0 < = y1 < y2``.
#'     output_size (Unionc(List[Tuple[int, int]], List[int])): size of the output
#'     sampling_ratio (int): sampling ratio for ROIAlign
#'     scales (Optional[List[float]]): If NULL, scales will be automatically inferred. Default is.null(value).
#'     mapper (Optional[LevelMapper]): If none, mapper will be automatically inferred. Default is.null(value).
#' Returns:
#'     result (Tensor)
multiscale_roi_align <- function(x_filtered,
                                 boxes,
                                 output_size,
                                 sampling_ratio,
                                 scales,
                                 mapper) {
  if (is.null(scales) || is.null(mapper)) {
    rlang::abort(      glue::glue("scales and mapper should not be NULL"))
  }

  num_levels <- length(x_filtered)
  rois <- convert_to_roi_format(boxes)

  if (num_levels == 1) {
    return(
      roi_align(),
      x_filtered[1],
      rois,
      output_size = output_size,
      spatial_scale = scales[1],
      sampling_ratio = sampling_ratio,
    )

    levels <- mapper(boxes)

    num_rois <- length(rois)
    num_channels <- x_filtered[1]$shape[2]

    dtype <- x_filtered[1]$dtype
    device <- x_filtered[1]$device
    result <-
      torch::torch_zeros(c(num_rois, num_channels) + output_size,
                         dtype = dtype,
                         device = device)

    tracing_results <- torch::torch_stack()
    # TODO need rework
    # for (level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales))) {
    for (i in seq_len(x_filtered)) {
      idx_in_level <- torch::torch_where(levels == level[[i]])[1]
      rois_per_level <- rois[idx_in_level]

      result_idx_in_level <- roi_align(
        scales[[i]][1],
        rois_per_level,
        output_size = output_size,
        spatial_scale = scales[[i]][2],
        sampling_ratio = sampling_ratio
      )

      # if (torchvision._is_tracing()) {
      #   tracing_results$append(result_idx_in_level$to(dtype))
      # } else {
        # result and result_idx_in_level's dtypes are based on dtypes of different
        # elements in x_filtered.  x_filtered contains tensors output by different
        # layers.  When autocast is active, it may choose different dtypes for
        # different layers' outputs.  Therefore, we defensively match result's dtype
        # before copying elements from result_idx_in_level in the following op.
        # We need to cast manually (can't rely on autocast to cast for us) because
        # the op acts on result in-place, and autocast only affects out-of-place ops.
        result[idx_in_level] <-
          result_idx_in_level$to(result$dtype)
      # }
    }

    # if (torchvision._is_tracing()) {
    #   result <- _onnx_merge_levels(levels, tracing_results)
    # }

    return(result)
  }


#' Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.
#'
#' It infers the scale of the pooling via the heuristics specified in eq. 1
#' of the `Feature Pyramid Network paper <https:%/%arxiv$org/abs/1612.03144>`_.
#' They keyword-only parameters ``canonical_scale`` and ``canonical_level``
#' correspond respectively to ``224`` and ``k0 = 4`` in eq. 1, and
#' have the following meaning: ``canonical_level`` is the target level of the pyramid from
#' which to pool a region of interest with ``w x h <- canonical_scale x canonical_scale``.
#'
#' Args:
#'     featmap_names (List[str]): the names of the feature maps that will be used
#'         for the pooling.
#'     output_size (Listc(Tuple[int, int]] or List[int)): output size for the pooled region
#'     sampling_ratio (int): sampling ratio for ROIAlign
#'     canonical_scale (int, optional): canonical_scale for LevelMapper
#'     canonical_level (int, optional): canonical_level for LevelMapper
#'
#' Examples::
#'
#'     >>> m <- torchvision$ops$MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
#'     >>> i <- OrderedDict()
#'     >>> i$feat1 <- torch::torch_rand(1, 5, 64, 64)
#'     >>> i$feat2 <- torch::torch_rand(1, 5, 32, 32)  # this feature won't be used in the pooling
#'     >>> i$feat3 <- torch::torch_rand(1, 5, 16, 16)
#'     >>> # create some random bounding boxes
#'     >>> boxes <- torch::torch_rand(6, 4) * 256; boxes[:, 2:] + = boxes[:, :2]
#'     >>> # original image size, before computing the feature maps
#'     >>> image_sizes <- [(512, 512)]
#'     >>> output <- m(i, [boxes], image_sizes)
#'     >>> print(output$shape)
#'     >>> torch::torch_Size([6, 5, 3, 3])
#'
MultiScaleRoIAlign <- torch::nn_module(
  "MultiScaleRoIAlign",
  annotations = list("scales", "map_levels"),

  initialize = function(
        self,
        featmap_names,
        output_size,
        sampling_ratio,
        canonical_scale = 224,
        canonical_level = 4) {

        # _log_api_usage_once(self)
        if is.integer(output_size) {
            output_size <- (output_size, output_size)
        }
        self$featmap_names <- featmap_names
        self$sampling_ratio <- sampling_ratio
        self$output_size <- tuple(output_size)
        self$scales <- NULL
        self$map_levels <- NULL
        self$canonical_scale <- canonical_scale
        self$canonical_level <- canonical_level

        }

  # Args:
  #     x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
  #         all the same number of channels, but they can have different sizes.
  #     boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
  #         (x1, y1, x2, y2) format and in the image reference size, not the feature map
  #         reference. The coordinate must satisfy ``0 < = x1 < x2`` and ``0 < = y1 < y2``.
  #     image_shapes (Listc(Tuple[height, width])): the sizes of each image before they
  #         have been fed to a CNN to obtain feature maps. This allows us to infer the
  #         scale factor for each one of the levels to be pooled.
  # Returns:
  #     result (Tensor)
  forward = function(
        self,
        x,
        boxes,
        image_shapes) {

        x_filtered <- _filter_input(x, self$featmap_names)
        if (self$is.null(scales) or self$is.null(map_levels)) {
            self$scales, self$map_levels <- setup_scales(
                x_filtered, image_shapes, self$canonical_scale, self$canonical_level
            )
        }

        return(multiscale_roi_align()
            x_filtered,
            boxes,
            self$output_size,
            self$sampling_ratio,
            self$scales,
            self$map_levels,
        )

  }

__repr__ = function(self) -> str:
        return(()
            f"list(self.__class__.__name__}(featmap_names = {self$featmap_names), "
            f"output_size = list(self$output_size}, sampling_ratio = {self$sampling_ratio))"
        )

}
