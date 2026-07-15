#include <cstdio>
#include <vector>

#include "check.hpp"
#include "device_buffer.hpp"
#include "vector_add.hpp"

int main() {
  constexpr int n = 1 << 20;

  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i);
  }

  try {
    DeviceBuffer<float> da(n), db(n), dc(n);
    da.from_host(ha.data());
    db.from_host(hb.data());
    CudaCheck(VectorAdd(da.get(), db.get(), dc.get(), n));
    CudaCheck(cudaDeviceSynchronize());
    dc.to_host(hc.data());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  std::printf("vector_add ok: c[0]=%.1f c[%d]=%.1f\n", hc[0], n - 1, hc[n - 1]);
  return 0;
}
