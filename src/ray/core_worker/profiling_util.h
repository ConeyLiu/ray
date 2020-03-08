#ifndef RAY_CORE_WORKER_PROFILING_UTIL_H
#define RAY_CORE_WORKER_PROFILING_UTIL_H

#include "profiling.h"

namespace ray {

namespace worker {

class ProfilingUtil {
public:
  static void initialize(const std::shared_ptr<Profiler> profiler) {
    if (profiler_ == nullptr) {
      profiler_ = profiler;
    }
  }

  static std::unique_ptr<worker::ProfileEvent> CreateProfileEvent(
      const std::string &event_type) {
    return std::unique_ptr<worker::ProfileEvent>(
      new worker::ProfileEvent(profiler_, event_type));
  }

private:
   static std::shared_ptr<Profiler> profiler_;
};

} // end namespace::worker
} // end namespace::ray
#endif