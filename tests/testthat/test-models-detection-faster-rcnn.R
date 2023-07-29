test_that("backbone_with_FPN works", {
  model <- model_resnet34()

  expect_no_error(backbone_with_fpn(model))
  # expect_equal_to_r(out[1,1], -1.1959798336029053, tolerance = 1e-5) # value taken from pytorch
})

