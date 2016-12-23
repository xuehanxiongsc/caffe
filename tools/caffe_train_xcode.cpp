#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(sigint_effect, "stop",
              "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
              "Optional; action to take when a SIGHUP signal is received: "
              "snapshot, stop or none.");

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
                                             const std::string& flag_value) {
    if (flag_value == "stop") {
        return caffe::SolverAction::STOP;
    }
    if (flag_value == "snapshot") {
        return caffe::SolverAction::SNAPSHOT;
    }
    if (flag_value == "none") {
        return caffe::SolverAction::NONE;
    }
    LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
    const std::string solver_file("/Users/xuehan.xiong/framework/dot/python/cpm8_enet_solver.prototxt");
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(solver_file, &solver_param);
    
    Caffe::set_mode(Caffe::CPU);
    caffe::SignalHandler signal_handler(GetRequestedAction(FLAGS_sigint_effect),
                                        GetRequestedAction(FLAGS_sighup_effect));
    
    shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    solver->SetActionFunction(signal_handler.GetActionFunction());
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
    LOG(INFO) << "Optimization Done.";
    return 0;
}

int main(int argc, char** argv) {
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Set version
    gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
    
    train();
    
    return 0;
}
