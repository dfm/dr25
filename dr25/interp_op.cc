#include <iostream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Interp")
  .Attr("T: {float, double}")
  .Attr("check_sorted: bool = true")
  .Input("t: T")
  .Input("x: T")
  .Input("y: T")
  .Output("z: T")
  .Output("dz: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle t, x, y, y0;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &y));

    TF_RETURN_IF_ERROR(c->Concatenate(t, x, &y0));
    TF_RETURN_IF_ERROR(c->Merge(y, y0, &y));

    c->set_output(0, t);
    c->set_output(1, t);
    return Status::OK();
  });

template <typename T>
class InterpOp : public OpKernel {
 public:
  explicit InterpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("check_sorted", &check_sorted_));
  }

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor = context->input(0);
    const Tensor& x_tensor = context->input(1);
    const Tensor& y_tensor = context->input(2);

    OP_REQUIRES(context, (t_tensor.dims() == 1), errors::InvalidArgument("'t' must be 1-dimensional"));
    OP_REQUIRES(context, (x_tensor.dims() == 1), errors::InvalidArgument("'x' must be 1-dimensional"));
    OP_REQUIRES(context, (y_tensor.dims() == 2), errors::InvalidArgument("'Y' must be 2-dimensional"));

    // Dimensions
    const int64 N = x_tensor.dim_size(0);
    const int64 M = t_tensor.dim_size(0);
    OP_REQUIRES(context, (y_tensor.dim_size(0) == M), errors::InvalidArgument("'Y' must have shape (M, N)"));
    OP_REQUIRES(context, (y_tensor.dim_size(1) == N), errors::InvalidArgument("'Y' must have shape (M, N)"));

    // Access the data
    const auto t = t_tensor.template flat<T>();
    const auto x = x_tensor.template flat<T>();
    const auto y = y_tensor.template matrix<T>();

    // Check for sorted order
    if (check_sorted_) {
      for (int64 n = 0; n < N-1; ++n)
        OP_REQUIRES(context, (x(n+1) > x(n)), errors::InvalidArgument("'x' must be sorted"));
    }

    // Output
    Tensor* z_tensor = NULL;
    Tensor* dz_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_tensor.shape(), &z_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, t_tensor.shape(), &dz_tensor));
    auto z = z_tensor->template flat<T>();
    auto dz = dz_tensor->template flat<T>();

    for (int64 m = 0; m < M; ++m) {
      auto value = t(m);
      if (value <= x(0)) {
        dz(m) = 0.0;
        z(m) = y(m, 0);
        continue;
      }
      if (value >= x(N-1)) {
        dz(m) = 0.0;
        z(m) = y(m, N-1);
        continue;
      }
      int64 left = 0, right = N-1;
      while (left < right) {
        int64 middle = left + ((right - left) >> 1);
        if (x(middle) < value) {
          left = middle + 1;
        } else {
          right = middle;
        }
      }
      left = right - 1;
      dz(m) = (y(m, right) - y(m, left)) / (x(right) - x(left));
      z(m) = (value - x(left)) * dz(m) + y(m, left);
    }
  }
 private:
  bool check_sorted_;
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Interp").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      InterpOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
