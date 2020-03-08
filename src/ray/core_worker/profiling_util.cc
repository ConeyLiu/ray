#include "profiling_util.h"

namespace ray {

namespace worker {
  std::shared_ptr<Profiler> ProfilingUtil::profiler_ = nullptr;

} // ending ray

} // ending worker
