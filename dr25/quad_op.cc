#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <limits>

#include "quad.h"

using namespace tensorflow;

REGISTER_OP("Quad")
  .Attr("T: {float, double}")
  .Input("g1: T")
  .Input("g2: T")
  .Input("p: T")
  .Input("z: T")
  .Output("flux: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle s;
    shape_inference::DimensionHandle d;
    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(s, 0), c->Dim(c->input(3), 0), &d));
    c->set_output(0, c->input(3));
    return Status::OK();
  });

template <typename T>
class QuadOp : public OpKernel {
 public:
  explicit QuadOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& g1_tensor = context->input(0);
    const Tensor& g2_tensor = context->input(1);
    const Tensor& p_tensor = context->input(2);
    const Tensor& z_tensor = context->input(3);

    // Dimensions
    int64 N = g1_tensor.NumElements();
    int64 M = 1;
    if (z_tensor.dims() > g1_tensor.dims()) {
      OP_REQUIRES(context, (z_tensor.dims() == g1_tensor.dims() + 1), errors::InvalidArgument("invalid dimensions"));
      M = z_tensor.dim_size(z_tensor.dims() - 1);
    }
    OP_REQUIRES(context, (g2_tensor.NumElements() == N), errors::InvalidArgument("all inputs must have matching shapes"));
    OP_REQUIRES(context, (p_tensor.NumElements() == N), errors::InvalidArgument("all inputs must have matching shapes"));
    OP_REQUIRES(context, (z_tensor.NumElements() == N * M), errors::InvalidArgument("all inputs must have matching shapes"));

    // Output
    Tensor* flux_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, z_tensor.shape(), &flux_tensor));

    // Access the data
    const auto g1 = g1_tensor.template flat<T>();
    const auto g2 = g2_tensor.template flat<T>();
    const auto p = p_tensor.template flat<T>();
    const auto z = z_tensor.template flat<T>();
    auto flux = flux_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) {
      for (int64 m = 0; m < M; ++m) {
        flux(n) = batman::quad<T>(g1(n), g2(n), p(n), z(n * M + m));
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Quad").Device(DEVICE_CPU).TypeConstraint<type>("T"),              \
      QuadOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
