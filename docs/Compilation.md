# Compiling USearch

## Build

Before building the first time, please pull submodules.

```sh
git submodule update --init --recursive
```

### Linux

```sh
cmake -B ./build_release \
    -DCMAKE_CXX_COMPILER="g++-12" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_JEMALLOC=1 && \
    make -C ./build_release -j

# To benchmark in C++:
./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin

# Large benchmark on a 1TB dataset:
./build_release/bench \
    --vectors datasets/t2i_1B/base.1B.fbin \
    --queries datasets/t2i_1B/query.public.100K.fbin \
    --neighbors datasets/t2i_1B/groundtruth.public.100K.ibin \
    --output datasets/t2i_1B/index.usearch \
    --cos

# To benchmark in Python:
python python/bench.py \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin
```

### MacOS

```sh
brew install libomp llvm
cmake \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DUSEARCH_USE_SIMD=1 \
    -DUSEARCH_USE_OPENMP=1 \
     -B ./build_release && \
    make -C ./build_release -j 
```

### Python

```sh
pip install -e .
pytest python/test.py
```

### JavaScript

```sh
npm install
node javascript/test.js
npm publish
```

### Rust

```sh
cargo test -p usearch
cargo publish
```

### Java

```sh
gradle clean build
java -cp . -Djava.library.path=/Users/av/github/usearch/build/libs/usearch/shared java/cloud/unum/usearch/Index.java
```

Or step by-step:

```sh
cs java/unum/cloud/usearch
javac -h . Index.java

# Ubuntu:
g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -fPIC -o libusearch.so cloud_unum_usearch_Index.o -lc

# Windows
g++ -c -I%JAVA_HOME%\include -I%JAVA_HOME%\include\win32 cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -o USearchJNI.dll cloud_unum_usearch_Index.o -Wl,--add-stdcall-alias

# MacOS
g++ -std=c++11 -c -fPIC \
    -I../../../../include -I../../../../src -I../../../../fp16/include -I../../../../simsimd/include \
    -I${JAVA_HOME}/include -I${JAVA_HOME}/include/darwin cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -dynamiclib -o libusearch.dylib cloud_unum_usearch_Index.o -lc
# Run linking to that directory
java -cp . -Djava.library.path=/Users/av/github/usearch/java/cloud/unum/usearch/ Index.java
java -cp . -Djava.library.path=/Users/av/github/usearch/java cloud.unum.usearch.Index
```

### Wolfram

```sh
brew install --cask wolfram-engine
```