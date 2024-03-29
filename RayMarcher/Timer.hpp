#ifndef RAYMARCHER_TIMER_HPP
#define RAYMARCHER_TIMER_HPP

#include <chrono>
#include <type_traits>

namespace rmcuda
{
// adapted from https://gist.github.com/gongzhitaao/7062087

/**
* Timer using high precision clock.
* 
* @tparam T Type representing time. Must be floating point.
*/
template <typename T>
class Timer
{
  static_assert(std::is_floating_point<T>::value,
    "Timer must use a floating point type");
public:
  Timer() : m_begin(clock_t::now()) {}

  /**
  * Resets the timer to zero.
  */
  void reset() { m_begin = clock_t::now(); }

  /**
  * Gets the elapsed time.
  * 
  * @return Elapsed time.
  */
  T elapsed() const
  {
    return std::chrono::duration_cast<second_t>
      (clock_t::now() - m_begin).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_t;
  typedef std::chrono::duration<T, std::ratio<1>> second_t;
  std::chrono::time_point<clock_t> m_begin;
};
} // namespace rmcuda

#endif // !RAYMARCHER_TIMER_HPP