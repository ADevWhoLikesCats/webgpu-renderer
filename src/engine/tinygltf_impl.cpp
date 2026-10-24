// tinygltf implementation TU. Exactly one file in the entire build
// defines TINYGLTF_IMPLEMENTATION, which pulls in tinygltf's compiled
// symbols. All other files that #include <tiny_gltf.h> just get the
// declarations.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include "tiny_gltf.h"
