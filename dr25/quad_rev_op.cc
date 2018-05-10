#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <limits>

#include "quad.h"
#include "ellint_grad.h"

using namespace tensorflow;

REGISTER_OP("QuadRev")
  .Attr("T: {float, double}")
  .Input("g1: T")
  .Input("g2: T")
  .Input("p: T")
  .Input("z: T")
  .Input("bflux: T")
  .Output("bg1: T")
  .Output("bg2: T")
  .Output("bp: T")
  .Output("bz: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle s;
    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(3), &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(4), &s));
    c->set_output(0, s);
    c->set_output(1, s);
    c->set_output(2, s);
    c->set_output(3, s);
    return Status::OK();
  });

template <typename T>
class QuadRevOp : public OpKernel {
 public:
  explicit QuadRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& g1_tensor = context->input(0);
    const Tensor& g2_tensor = context->input(1);
    const Tensor& p_tensor = context->input(2);
    const Tensor& z_tensor = context->input(3);
    const Tensor& bflux_tensor = context->input(4);

    // Dimensions
    const int64 N = g1_tensor.NumElements();
    OP_REQUIRES(context, (g2_tensor.NumElements() == N), errors::InvalidArgument("all inputs must have matching shapes"));
    OP_REQUIRES(context, (p_tensor.NumElements() == N), errors::InvalidArgument("all inputs must have matching shapes"));
    OP_REQUIRES(context, (z_tensor.NumElements() == N), errors::InvalidArgument("all inputs must have matching shapes"));
    OP_REQUIRES(context, (bflux_tensor.NumElements() == N), errors::InvalidArgument("all inputs must have matching shapes"));

    // Output
    Tensor* bg1_tensor = NULL;
    Tensor* bg2_tensor = NULL;
    Tensor* bp_tensor = NULL;
    Tensor* bz_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, z_tensor.shape(), &bg1_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, z_tensor.shape(), &bg2_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, z_tensor.shape(), &bp_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, z_tensor.shape(), &bz_tensor));

    // Access the data
    const auto g1 = g1_tensor.template flat<T>();
    const auto g2 = g2_tensor.template flat<T>();
    const auto p = p_tensor.template flat<T>();
    const auto z = z_tensor.template flat<T>();
    const auto bflux = bflux_tensor.template flat<T>();
    auto bg1 = bg1_tensor->template flat<T>();
    auto bg2 = bg2_tensor->template flat<T>();
    auto bp = bp_tensor->template flat<T>();
    auto bz = bz_tensor->template flat<T>();

    typedef Eigen::Matrix<T, 4, 1> DerType;

    for (int64 n = 0; n < N; ++n) {
      Eigen::AutoDiffScalar<DerType> ad_g1(g1(n), 4, 0),
                                     ad_g2(g2(n), 4, 1),
                                     ad_p(p(n), 4, 2),
                                     ad_z(z(n), 4, 3);
      auto f = batman::quad(ad_g1, ad_g2, ad_p, ad_z);
      auto d = f.derivatives();
      bg1(n) = bflux(n) * d(0);
      bg2(n) = bflux(n) * d(1);
      bp(n) = bflux(n) * d(2);
      bz(n) = bflux(n) * d(3);
    }
  }
};


#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("QuadRev").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      QuadRevOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
