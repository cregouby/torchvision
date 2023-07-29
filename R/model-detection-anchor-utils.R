# from .image_list import ImageList


#' Module that generates anchors for a set of feature maps and
#' image sizes.
#'
#' The module support computing anchors at multiple sizes and aspect ratios
#' per feature map. This module assumes aspect ratio <- height / width for
#' each anchor.
#'
#' sizes and aspect_ratios should have the same number of elements, and it should
#' correspond to the number of feature maps.
#'
#' sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
#' and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
#' per spatial location for feature map i.
#'
#' Args:
#'     sizes (Tuple[Tuple[int]]) {
#'     aspect_ratios (Tuple[Tuple[float]]) {
AnchorGenerator <- torch::nn_module(
  "AnchorGenerator",
  annotations = {
        "cell_anchors"
    },
  initialize = function(
        self,
        sizes = c(c(128, 256, 512)),
        aspect_ratios = c(c(0.5, 1.0, 2.0))    ) {


        # if (!is.list(sizes[0])) {
        #     # TODO change this
        #     sizes <- tuple((s,) for s in sizes)
        if (!is.list(aspect_ratios[1])) {
            aspect_ratios <- rep(aspect_ratios, length(sizes))
        }

        self$sizes <- sizes
        self$aspect_ratios <- aspect_ratios
        # TODO bug Error in `purrr::map2()`:
        # ! Can't recycle `.x` (size 15) to match `.y` (size 45).
        self$cell_anchors <- purrr::map2(sizes, aspect_ratios, self$generate_anchors)

    # TODO: https:%/%github$com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self$scales, self$aspect_ratios)
    # This method assumes aspect_ratio = height / width for an anchor.
  },
  generate_anchors = function(
        self,
        scales,
        aspect_ratios,
        dtype = torch::torch_float32,
        device = torch::torch_device("cpu")
    ) {
        scales <- torch::torch_tensor(scales, dtype = dtype, device = device)
        aspect_ratios <- torch::torch_tensor(aspect_ratios, dtype = dtype, device = device)
        h_ratios <- torch::torch_sqrt(aspect_ratios)
        w_ratios <- 1 / h_ratios

        ws <- (w_ratios[, NULL] * scales[NULL, ])$view(-1)
        hs <- (h_ratios[, NULL] * scales[NULL, ])$view(-1)

        base_anchors <- torch::torch_stack(c(-ws, -hs, ws, hs), dim = 1) / 2
        return(base_anchors$round())

  },
  set_cell_anchors = function(self, dtype, device) {
        self$cell_anchors <- purrr::map(self$cell_anchors, ~.x$to(dtype = dtype, device = device))

  },
  num_anchors_per_location = function(self) {
        return(purrr::map2(self$sizes, self$aspect_ratios, ~length(.x) * length(.y)))
  },

  # For every combination of (a, (g, s), i) in (self$cell_anchors, zip(grid_sizes, strides), 0:2),
  # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
  grid_anchors = function(self, grid_sizes, strides) {
        anchors <- list()
        cell_anchors <- self$cell_anchors
        if (is.null(cell_anchors)) {
          rlang::abort(glue::glue("cell_anchors should not be NULL"))
        }
        if (length(grid_sizes) != length(strides) || length(strides) != length(cell_anchors)) {
          rlang::abort(glue::glue("Anchors should be Tuple[Tuple[int]] because each feature ",
                                  "map could potentially have different sizes and aspect ratios. ",
                                  "There needs to be a match between the number of ",
                                  "feature maps passed and the number of sizes / aspect ratios specified."))
        }
        for (i in seq_len(grid_sizes)) {
            grid_height <- grid_sizes[[i]][1]
            grid_width <- grid_sizes[[i]][2]
            stride_height <- strides[[i]][1]
            stride_width <- strides[[i]][2]
            base_anchors <- cell_anchors[[i]]
            device <- base_anchors$device

            # For output anchor, compute c(x_center, y_center, x_center, y_center)
            shifts_x <- torch::torch_arange(1, grid_width, dtype = torch::torch_int32, device = device) * stride_width
            shifts_y <- torch::torch_arange(1, grid_height, dtype = torch::torch_int32, device = device) * stride_height
            shift_xy <- torch::torch_meshgrid(list(shifts_y, shifts_x), indexing = "ij")
            shift_x <- shift_xy[[1]]$reshape(-1)
            shift_y <- shift_xy[[2]]$reshape(-1)
            shifts <- torch::torch_stack(list(shift_x, shift_y, shift_x, shift_y), dim = 1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors$append((shifts$view(-1, 1, 4) + base_anchors$view(1, -1, 4))$reshape(-1, 4))
        }

        return(anchors)

  },
  forward = function(self, image_list, feature_maps) {
        grid_sizes <- purrr::map(feature_maps, ~tail(.x$shape,2))
        image_size <- tail(image_list$tensors$shape, 2)
        dtype <- feature_maps[0]$dtype
        device <- feature_maps[0]$device
        strides <- purrr::map(grid_sizes, ~c(
                torch::torch_empty(dtype = torch::torch_int64, device = device)$fill_(image_size[1] %/% .x[1]),
                torch::torch_empty(dtype = torch::torch_int64, device = device)$fill_(image_size[2] %/% .x[2])
        ))
        self$set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps <- self$grid_anchors(grid_sizes, strides)
        anchors <- torch::torch_cat(rep(anchors_over_all_feature_maps, length(image_list$image_sizes)))
        # for _ in range(length()) {
        #     anchors_in_image <- [anchors_per_feature_map for anchors_per_feature_map in ]
        #     anchors$append(anchors_in_image)
        # anchors <- [torch::torch_cat(anchors_per_image) for anchors_per_image in anchors]
        return(anchors)
        }
)


#' This module generates the default boxes of SSD for a set of feature maps and image sizes.
#'
#' Args:
#'     aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
#'     min_ratio (float): The minimum scale :math:`\textlist(s}_{\text{min})` of the default boxes used in the estimation
#'         of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
#'     max_ratio (float): The maximum scale :math:`\textlist(s}_{\text{max})`  of the default boxes used in the estimation
#'         of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
#'     scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
#'         the ``min_ratio`` and ``max_ratio`` parameters.
#'     steps (List[int]], optional): It's a hyper-parameter that affects the tiling of default boxes. If not provided
#'         it will be estimated from the data.
#'     clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
#'         is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
DefaultBoxGenerator <- torch::nn_module(
  "DefaultBoxGenerator",
  initialize = function(
        self,
        aspect_ratios,
        min_ratio = 0.15,
        max_ratio = 0.9,
        scales = NULL,
        steps = NULL,
        clip = TRUE    ) {

        if (!is.null(steps) && length(aspect_ratios) != length(steps)) {
            rlang::abort(glue::glue("aspect_ratios and steps should have the same length"))
        self$aspect_ratios <- aspect_ratios
        self$steps <- steps
        self$clip <- clip
        num_outputs <- length(aspect_ratios)

        # Estimation of default boxes scales
        if (is.null(scales)) {
            if (num_outputs > 1) {
                range_ratio <- max_ratio - min_ratio
                self$scales <- min_ratio + range_ratio * seq(num_outputs) / (num_outputs - 1)
                self$scales$append(1)
            } else {
                self$scales <- c(min_ratio, max_ratio)
            }
        } else {
            self$scales <- scales
        }

        self._wh_pairs <- self._generate_wh_pairs(num_outputs)
        }
  },
  generate_wh_pairs = function(
        self, num_outputs, dtype = torch::torch_float32, device = torch::torch_device("cpu")
    ) {
        wh__pairs <- torch::torch_tensor(NULL)
        for (k in range(num_outputs)) {
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k <- self$scales[k]
            s_prime_k <- sqrt(self$scales[k] * self$scales[k + 1])
            wh_pairs <- list(c(s_k, s_k), c(s_prime_k, s_prime_k))

            # Adding 2 pairs for each aspect ratio of the feature map k
            for (ar in self$aspect_ratios[k]) {
                sq_ar <- sqrt(ar)
                w <- self$scales[k] * sq_ar
                h <- self$scales[k] / sq_ar
                append(wh_pairs, c(c(w, h), c(h, w)))
            }

            wh__pairs$append(torch::torch_tensor(wh_pairs, dtype = dtype, device = device))
        }
        return(wh__pairs)
  },
  num_anchors_per_location = function(self) {
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feature map.
        return(purrr::map(self$aspect_ratios, ~2 + 2 * length(.x)))

  },
  # Default Boxes calculation based on page 6 of SSD paper
  grid_default_boxes = function(
        self, grid_sizes, image_size, dtype = torch::torch_float32    ) {
        default_boxes <- torch::torch_tensor(NULL)
        for (k in seq_len(grid_sizes)) {
            # Now add the default boxes for each width-height pair
            if (!is.null(self$steps)) {
                x_f_k <- image_size[2] / self$steps[k]
                y_f_k <- image_size[1] / self$steps[k]
            } else {
                list(y_f_k, x_f_k) %<-% grid_sizes[[k]]
            }

            shifts_x <- ((torch::torch_arange(1, grid_sizes[[k]][2]) + 0.5) / x_f_k)$to(dtype = dtype)
            shifts_y <- ((torch::torch_arange(1, grid_sizes[[k]][1]) + 0.5) / y_f_k)$to(dtype = dtype)
            list(shift_y, shift_x) %<-% torch::torch_meshgrid(list(shifts_y, shifts_x), indexing = "ij")
            shift_x <- shift_x$reshape(-1)
            shift_y <- shift_y$reshape(-1)

            shifts <- torch::torch_stack(c(shift_x, shift_y) * length(self._wh_pairs[k]), dim = -1)$reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            if (self$clip) {
              wh_pair <- self._wh_pairs[k]$clamp(min = 0, max = 1)
            }  else {
              wh_pair <-   self._wh_pairs[k]
            }
            # TODO need rework
            wh_pairs <- wh_pair$`repeat`((grid_sizes[[k]][1] * grid_sizes[[k]][1]), 2)

            default_box <- torch::torch_cat(list(shifts, wh_pairs), dim = 2)

            default_boxes$append(default_box)
        }

        return(torch::torch_cat(default_boxes, dim = 1))

  },
  repr = function(self) {
        s <- glue::glue(
            "{self.__class__.__name__}(",
            "aspect_ratios = {self$aspect_ratios}",
            ", clip = {self$clip}",
            ", scales = {self$scales}",
            ", steps = {self$steps}",
            ")"
        )
        return(s)

  },
  forward = function(self, image_list, feature_maps) {
        grid_sizes <- purrr::map(feature_maps, ~tail(.x$shape,2))
        image_size <- tail(image_list$shape,2)
        dtype <- feature_maps[1]$dtype
        device <- feature_maps[1]$device
        default_boxes <- self._grid_default_boxes(grid_sizes, image_size, dtype = dtype)
        default_boxes <- default_boxes$to(device)

        x_y_size <- torch::torch_tensor(c(image_size[2], image_size[1]), device = default_boxes$device)
        dboxes <- torch::torch_stack(append(
          default_boxes,
          rep(torch::torch_cat(
            list(
              (dboxes_in_image[, 1:3] - 0.5 * dboxes_in_image[, 3:N]) * x_y_size,
              (dboxes_in_image[, 1:3] + 0.5 * dboxes_in_image[, 3:N]) * x_y_size,
            ),
            -1
          ), length(image_list$image_sizes))

        ))
        return(dboxes)
  }
)
