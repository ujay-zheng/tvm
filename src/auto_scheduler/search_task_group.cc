#include <tvm/runtime/registry.h>
#include <tvm/auto_scheduler/search_task.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(SearchTaskGroupNode);

// SearchTaskGroup::SearchTaskGroup(Array<SearchTask> tasks, Array<Array<Integer> > launch_id_list){
//     auto node = make_object<SearchTaskGroupNode>();
//     node->tasks = std::move(tasks);
//     node->launch_id_list = std::move(launch_id_list);
//     data_ = std::move(node);
// }

// TVM_REGISTER_GLOBAL("auto_scheduler.SearchTaskGroup")
//     .set_body_typed([](Array<SearchTask> tasks, Array<Array<Integer> > launch_id_list){
//         return SearchTaskGroup(tasks, launch_id_list);
//     });

SearchTaskGroup::SearchTaskGroup(Array<SearchTask> tasks, Array<Array<Integer> > topological_seq, int streams_num, int events_num){
    auto node = make_object<SearchTaskGroupNode>();
    node->tasks = std::move(tasks);
    node->topological_seq = std::move(topological_seq);
    node->streams_num = streams_num;
    node->events_num = events_num;
    data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("auto_scheduler.SearchTaskGroup")
    .set_body_typed([](Array<SearchTask> tasks, Array<Array<Integer> > topological_seq, int streams_num, int events_num){
        return SearchTaskGroup(tasks, topological_seq, streams_num, events_num);
    });

}
}