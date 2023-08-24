// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformation_helper.hpp"

#include "log/debug.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace helper {

using namespace ov::opset10;

void GetConvData(std::shared_ptr<ngraph::op::ConvolutionIE> conv, ConvData& conv_data) {
    OPENVINO_ASSERT(conv);
    conv_data.output_height = conv->get_output_shape(0)[2];
    conv_data.output_width = conv->get_output_shape(0)[3];
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.filter_count = conv->input_value(1).get_shape()[0];
    conv_data.filter_channel_count = conv->input_value(1).get_shape()[1];
    conv_data.filter_height = conv->input_value(1).get_shape()[2];
    conv_data.filter_width = conv->input_value(1).get_shape()[3];
    conv_data.filter_dilation_height = conv->get_dilations()[0];
    conv_data.filter_dilation_width = conv->get_dilations()[1];
    conv_data.filter_stride_height = conv->get_strides()[0];
    conv_data.filter_stride_width = conv->get_strides()[1];
    conv_data.output_channel_count = conv_data.filter_count;
    conv_data.pads_begin_height = conv->get_pads_begin()[0];
    conv_data.pads_begin_width = conv->get_pads_begin()[1];
    conv_data.pads_end_height = conv->get_pads_end()[0];
    conv_data.pads_end_width = conv->get_pads_end()[1];
    conv_data.padding_type = conv->get_auto_pad();
    conv_data.element_type = conv->get_element_type();
}

void GetConvData(std::shared_ptr<Convolution> conv, ConvData& conv_data) {
    OPENVINO_ASSERT(conv);
    conv_data.output_height = conv->get_output_shape(0)[2];
    conv_data.output_width = conv->get_output_shape(0)[3];
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.filter_count = conv->input_value(1).get_shape()[0];
    conv_data.filter_channel_count = conv->input_value(1).get_shape()[1];
    conv_data.filter_height = conv->input_value(1).get_shape()[2];
    conv_data.filter_width = conv->input_value(1).get_shape()[3];
    conv_data.filter_dilation_height = conv->get_dilations()[0];
    conv_data.filter_dilation_width = conv->get_dilations()[1];
    conv_data.filter_stride_height = conv->get_strides()[0];
    conv_data.filter_stride_width = conv->get_strides()[1];
    conv_data.output_channel_count = conv_data.filter_count;
    conv_data.pads_begin_height = conv->get_pads_begin()[0];
    conv_data.pads_begin_width = conv->get_pads_begin()[1];
    conv_data.pads_end_height = conv->get_pads_end()[0];
    conv_data.pads_end_width = conv->get_pads_end()[1];
    conv_data.padding_type = conv->get_auto_pad();
    conv_data.element_type = conv->get_element_type();
}

std::function<bool(ov::Output<ov::Node>)> consumers_and_rank(const size_t expected_count,
                                                             const ov::Dimension& expected_rank) {
    return [=](ov::Output<ov::Node> output) -> bool {
        return ov::pass::pattern::consumers_count(expected_count)(output) &&
               ov::pass::pattern::rank_equals(expected_rank)(output);
    };
}

bool TransposeOrderMatches(std::shared_ptr<Transpose> transpose, std::vector<size_t> order) {
    if (!transpose)
        return false;
    const ov::Output<ov::Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values = std::dynamic_pointer_cast<Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;

    const auto data = const_with_order_values->cast_vector<size_t>();
    if (data.empty())
        return false;

    if (!std::equal(order.begin(), order.end(), data.begin()))
        return false;

    return true;
}

std::shared_ptr<StridedSlice> FlatCrop(ov::Output<ov::Node> input, size_t offset, size_t size) {
    return std::make_shared<StridedSlice>(
        input,                                                                         // data
        Constant::create(ov::element::i64, ov::Shape{2}, {(size_t)0, offset}),         // begin sice index
        Constant::create(ov::element::i64, ov::Shape{2}, {(size_t)0, offset + size}),  // end slice index
        Constant::create(ov::element::i64, ov::Shape{2}, {(size_t)1, (size_t)1}),      // strides
        std::vector<int64_t>{1, 0},                                                    // begin mask
        std::vector<int64_t>{1, 0});                                                   // end mask
}

std::shared_ptr<ov::Node> VerifyBiasGetConst(std::shared_ptr<ov::Node> conv, std::shared_ptr<ov::Node> add) {
    auto add_const = std::dynamic_pointer_cast<Constant>(add->input_value(1).get_node_shared_ptr());

    // Check if it's really a bias and not just addition
    if (add_const) {
        auto bias_size = shape_size(add_const->get_shape());
        auto conv_filter_count = conv->get_output_shape(0)[1];
        if (bias_size == conv_filter_count)
            return add_const;
    }
    return nullptr;
}

std::shared_ptr<ov::Node> InsertFQLayer(const std::shared_ptr<FakeQuantize> fq_layer,
                                        std::shared_ptr<ov::Node> last_node) {
    if (fq_layer != nullptr) {
        auto new_fq = fq_layer->clone_with_new_inputs(
            {last_node,
             Constant::create(ov::element::f32,
                              ov::Shape{1},
                              std::dynamic_pointer_cast<Constant>(fq_layer->input_value(1).get_node_shared_ptr())
                                  ->cast_vector<float>()),
             Constant::create(ov::element::f32,
                              ov::Shape{1},
                              std::dynamic_pointer_cast<Constant>(fq_layer->input_value(2).get_node_shared_ptr())
                                  ->cast_vector<float>()),
             Constant::create(ov::element::f32,
                              ov::Shape{1},
                              std::dynamic_pointer_cast<Constant>(fq_layer->input_value(3).get_node_shared_ptr())
                                  ->cast_vector<float>()),
             Constant::create(ov::element::f32,
                              ov::Shape{1},
                              std::dynamic_pointer_cast<Constant>(fq_layer->input_value(4).get_node_shared_ptr())
                                  ->cast_vector<float>())});
        copy_runtime_info(new_fq, fq_layer);
        return new_fq;
    }
    return last_node;
}

void remove_single_input_node(std::shared_ptr<ov::Node> node) {
    const ov::Shape input_node_shape = node->get_input_shape(0);
    const ov::Shape output_node_shape = node->get_output_shape(0);

    std::shared_ptr<ov::Node> node_parent = node->get_input_node_shared_ptr(0);
    if (!node_parent) {
        THROW_GNA_EXCEPTION << "The removing node has no parrent node";
    }
    if (!std::equal(input_node_shape.begin(), input_node_shape.end(), output_node_shape.begin())) {
        auto reshape_const_node =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{output_node_shape.size()}, output_node_shape);
        node_parent = std::make_shared<Reshape>(node_parent, reshape_const_node, false);
    }

    ov::replace_output_update_name(node->output(0), node_parent->output(0));
}

}  // namespace helper
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
