# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build"

# Include any dependencies generated for this target.
include test/CMakeFiles/changename.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/changename.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/changename.dir/flags.make

test/CMakeFiles/changename.dir/changename.cpp.o: test/CMakeFiles/changename.dir/flags.make
test/CMakeFiles/changename.dir/changename.cpp.o: ../test/changename.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/changename.dir/changename.cpp.o"
	cd "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test" && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/changename.dir/changename.cpp.o -c "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/test/changename.cpp"

test/CMakeFiles/changename.dir/changename.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/changename.dir/changename.cpp.i"
	cd "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test" && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/test/changename.cpp" > CMakeFiles/changename.dir/changename.cpp.i

test/CMakeFiles/changename.dir/changename.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/changename.dir/changename.cpp.s"
	cd "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test" && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/test/changename.cpp" -o CMakeFiles/changename.dir/changename.cpp.s

test/CMakeFiles/changename.dir/changename.cpp.o.requires:

.PHONY : test/CMakeFiles/changename.dir/changename.cpp.o.requires

test/CMakeFiles/changename.dir/changename.cpp.o.provides: test/CMakeFiles/changename.dir/changename.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/changename.dir/build.make test/CMakeFiles/changename.dir/changename.cpp.o.provides.build
.PHONY : test/CMakeFiles/changename.dir/changename.cpp.o.provides

test/CMakeFiles/changename.dir/changename.cpp.o.provides.build: test/CMakeFiles/changename.dir/changename.cpp.o


# Object files for target changename
changename_OBJECTS = \
"CMakeFiles/changename.dir/changename.cpp.o"

# External object files for target changename
changename_EXTERNAL_OBJECTS =

../bin/changename: test/CMakeFiles/changename.dir/changename.cpp.o
../bin/changename: test/CMakeFiles/changename.dir/build.make
../bin/changename: ../lib/libmylib.so
../bin/changename: /usr/local/lib/libopencv_cudabgsegm.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudaobjdetect.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudastereo.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_stitching.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudafeatures2d.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_superres.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudacodec.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_videostab.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudaoptflow.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudalegacy.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudawarping.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_aruco.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_bgsegm.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_bioinspired.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_ccalib.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_dpm.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_freetype.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_fuzzy.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_hdf.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_optflow.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_reg.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_saliency.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_stereo.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_structured_light.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_viz.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_rgbd.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_surface_matching.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_tracking.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_datasets.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_dnn.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_face.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_plot.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_text.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_shape.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_video.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_ximgproc.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_calib3d.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_features2d.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_flann.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_objdetect.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_ml.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_xphoto.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_highgui.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_photo.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudaimgproc.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudafilters.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudaarithm.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_videoio.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_imgproc.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_core.so.3.2.0
../bin/changename: /usr/local/lib/libopencv_cudev.so.3.2.0
../bin/changename: test/CMakeFiles/changename.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/changename"
	cd "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/changename.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/changename.dir/build: ../bin/changename

.PHONY : test/CMakeFiles/changename.dir/build

test/CMakeFiles/changename.dir/requires: test/CMakeFiles/changename.dir/changename.cpp.o.requires

.PHONY : test/CMakeFiles/changename.dir/requires

test/CMakeFiles/changename.dir/clean:
	cd "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test" && $(CMAKE_COMMAND) -P CMakeFiles/changename.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/changename.dir/clean

test/CMakeFiles/changename.dir/depend:
	cd "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch" "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/test" "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build" "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test" "/media/richardson/Richardson/陈亦雷/硕士/光场/程序/Modified Basline for LF Stitching/LF_Stitch/build/test/CMakeFiles/changename.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : test/CMakeFiles/changename.dir/depend
