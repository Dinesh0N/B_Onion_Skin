CXX = clang++
CXXFLAGS = -O3 -shared -fPIC -std=c++20 -Wall -Wextra -pthread -mavx2 -mfma -ffast-math

MINGW_CXX = /home/max/.local/mingw64/bin/x86_64-w64-mingw32-g++-posix
MINGW_FLAGS = -O3 -shared -std=c++20 -Wall -Wextra -mavx2 -mfma -ffast-math -static -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic

ZIG_CXX = uv tool run --from ziglang python-zig c++
LIPO = /usr/lib/llvm-22/bin/llvm-lipo

TARGET_LINUX = libonion_skin_core.so
TARGET_WIN = libonion_skin_core.dll
TARGET_WIN_ALT = onion_skin_core.dll
TARGET_MAC = libonion_skin_core.dylib

SRC = onion_skin_core.cpp fast_single.cpp
HDR = onion_skin_core.h thread_pool.hpp

all: linux windows macos

linux: $(TARGET_LINUX)

$(TARGET_LINUX): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET_LINUX)

windows: $(TARGET_WIN)

$(TARGET_WIN): $(SRC) $(HDR)
	$(MINGW_CXX) $(MINGW_FLAGS) $(SRC) -o $(TARGET_WIN)
	cp -f $(TARGET_WIN) $(TARGET_WIN_ALT)

macos: $(TARGET_MAC)

$(TARGET_MAC): $(SRC) $(HDR)
	$(ZIG_CXX) -target x86_64-macos.11.0 -O3 -shared -fPIC -std=c++20 -mavx2 -mfma -ffast-math $(SRC) -o libonion_skin_core_x86_64.dylib
	$(ZIG_CXX) -target aarch64-macos.11.0 -O3 -shared -fPIC -std=c++20 -ffast-math $(SRC) -o libonion_skin_core_arm64.dylib
	$(LIPO) -create libonion_skin_core_x86_64.dylib libonion_skin_core_arm64.dylib -output $(TARGET_MAC)
	rm -f libonion_skin_core_x86_64.dylib libonion_skin_core_arm64.dylib

clean:
	rm -f $(TARGET_LINUX) $(TARGET_WIN) $(TARGET_WIN_ALT) $(TARGET_MAC) libonion_skin_core_v*.so libonion_skin_core_v*.dll libonion_skin_core_v*.dylib libonion_skin_core_x86_64.dylib libonion_skin_core_arm64.dylib

.PHONY: all linux windows macos clean
