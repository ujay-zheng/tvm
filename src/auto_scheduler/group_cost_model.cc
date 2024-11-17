#include <tvm/auto_scheduler/cost_model.h>

namespace tvm {
namespace auto_scheduler {
    
TVM_REGISTER_OBJECT_TYPE(GroupCostModelNode);
TVM_REGISTER_OBJECT_TYPE(GroupRandomModelNode);
TVM_REGISTER_OBJECT_TYPE(GroupPythonBasedModelNode);

GroupRandomModel::GroupRandomModel() {
    ObjectPtr<GroupRandomModelNode> node = make_object<GroupRandomModelNode>();
    const auto* f = runtime::Registry::Get("auto_scheduler.cost_model.random_fill_float");
    ICHECK(f != nullptr);
    node->random_number_func = reinterpret_cast<const TypedPackedFunc<void(size_t, void*)>*>(f);
    data_ = std::move(node);
}

void GroupRandomModelNode::Update(const Array<Array<MeasureInput> >& inputs, 
                                    const Array<MeasureResult>& results) {}

void GroupRandomModelNode::Predict(const SearchTaskGroup& task_group, 
                                    const Array<Array<State> >& group_states,
                                    std::vector<float>* scores) {
    scores->resize(group_states.size());
    (*random_number_func)(group_states.size(), static_cast<void*>(scores->data()));
}

GroupPythonBasedModel::GroupPythonBasedModel(PackedFunc update_func, 
                                                PackedFunc predict_func) {
    auto node = make_object<GroupPythonBasedModelNode>();
    node->update_func = std::move(update_func);
    node->predict_func = std::move(predict_func);
    data_ = std::move(node);
}

void GroupPythonBasedModelNode::Update(const Array<Array<MeasureInput> >& inputs, 
                                        const Array<MeasureResult>& results) {
    update_func(inputs, results);
}

void GroupPythonBasedModelNode::Predict(const SearchTaskGroup& task_group, 
                                        const Array<Array<State> >& group_states, 
                                        std::vector<float>* scores) {
    scores->resize(group_states.size());
    predict_func(task_group, group_states, static_cast<void*>(scores->data()));
}

TVM_REGISTER_GLOBAL("auto_scheduler.GroupRandomModel").set_body_typed([]() {return GroupRandomModel(); });

TVM_REGISTER_GLOBAL("auto_scheduler.GroupPythonBasedModel")
    .set_body_typed([](PackedFunc update_func, PackedFunc predict_func) {
        return GroupPythonBasedModel(update_func, predict_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GroupCostModelUpdate")
    .set_body_typed([](GroupCostModel model, Array<Array<MeasureInput> > inputs, Array<MeasureResult> results) {
        model->Update(inputs, results);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GroupCostModelPredict")
    .set_body_typed([](GroupCostModel model, SearchTaskGroup task_group, Array<Array<State> > group_states) {
        std::vector<float> scores;
        model->Predict(task_group, group_states, &scores);
        Array<FloatImm> ret;
        for (auto x : scores) {
            ret.push_back(FloatImm(DataType::Float(32), x));
        }
        return ret;
    });

}
}