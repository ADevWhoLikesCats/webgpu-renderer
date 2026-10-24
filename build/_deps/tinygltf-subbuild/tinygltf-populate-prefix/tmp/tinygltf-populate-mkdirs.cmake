# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-src")
  file(MAKE_DIRECTORY "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-src")
endif()
file(MAKE_DIRECTORY
  "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-build"
  "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix"
  "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix/tmp"
  "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix/src/tinygltf-populate-stamp"
  "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix/src"
  "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix/src/tinygltf-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix/src/tinygltf-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/hrishikesh/Downloads/photonix-rewrite/build/_deps/tinygltf-subbuild/tinygltf-populate-prefix/src/tinygltf-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
