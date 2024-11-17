#include <tvm/runtime/registry.h>
#include <tvm/auto_scheduler/search_task.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(SearchTaskGroupNode);

SearchTaskGroup::SearchTaskGroup(Array<SearchTask> tasks, Array<Array<Integer> > launch_id_list){
    auto node = make_object<SearchTaskGroupNode>();
    node->tasks = std::move(tasks);
    node->launch_id_list = std::move(launch_id_list);
    data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("auto_scheduler.SearchTaskGroup")
    .set_body_typed([](Array<SearchTask> tasks, Array<Array<Integer> > launch_id_list){
        return SearchTaskGroup(tasks, launch_id_list);
    });

}
}