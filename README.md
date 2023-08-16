## Plugin Compilation



Assumptions
-----------
- OpenVINO version used is commit hash stored in `openvino_version.txt`
- The `openvino_version.txt` file located in plugin's source directory
- Commands syntax is bash-based
- There are following (environmental) variables defined:
````bash
export OPENVINO_SOURCE_DIR = <path to local openvino source directory>
export DEVELOPER_PACKAGE_DIR = <path to generated developer package directory>
export BUILD_TYPE = <Release|Debug|etc.>
export GNA_PLUGIN_SOURCE_DIR = <path to local gna plugin souce directory>
export GNA_PLUGIN_BUILD_DIR = <path to generated gna plugin build>
````



Prerequisties
-------------

##### Prepare OpenVino Developer Package

Switch to respective version of openvino
````bash
cd $OPENVINO_SOURCE_DIR
git checkout <commit hash from $GNA_PLUGIN_SOURCE_DIR/openvino_version.txt>
````

build it
````bash
cmake -S $OPENVINO_SOURCE_DIR -B $DEVELOPER_PACKAGE_DIR -D CMAKE_BUILD_TYPE=$BUILD_TYPE -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON
cmake --build $DEVELOPER_PACKAGE_DIR --target ov_dev_targets --config $BUILD_TYPE
````
> **Note**
> - It's important to keep in mind that `CMAKE_BUILD_TYPE` has no effect for multi-configuration generators such as Visual Studio. Therefore in this case config type has to be provided to build command as above.
> - There is yet another target `ie_dev_targets` can be used, but it seems to have identical result as `ov_dev_targets`

##### Get GNA local version of plugin source

Clone **gna.plugin** repository
````bash
git clone https://github.com/intel-innersource/frameworks.ai.gna.plugin.git $GNA_PLUGIN_SOURCE_DIR
````


##### Prepare configuration for functional tests
There is an additional step needed in order to run functional tests. You have create `plugins.xml` files (one for *Debug* and one for *Release*) with following content:
````xml
<ie>
    <plugins>
        <plugin name="GNA" location="${GNA_PLUGIN_BUILD_DIR}/bin/intel64/${BUILD_TYPE}/openvino_intel_gna_plugin.dll">
        </plugin>
    </plugins>
</ie>
````

and copy them respectively to `$OPENVINO_SOURCE_DIR/bin/intel64/$BUILD_TYPE/`

> **Note**
> Remember that Windows build uses different names of gna plugin library depending onf build type:
> - `openvino_intel_gna_plugin.dll` for `Release` build type
> - `openvino_intel_gna_plugind.dll` for `Debug` build type



Plugin build
------------

Create **gna.plugin** solution
````bash
cmake -S $GNA_PLUGIN_SOURCE_DIR -B $GNA_PLUGIN_BUILD_DIR -D OpenVINODeveloperPackage_DIR=$DEVELOPER_PACKAGE_DIR -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON
````

It happens often that for development purposes many versions of gna library exist in openvino `temp\` directory. Therefore it is possible to use specific version of the library by defining `GNA_LIB_VERSION` otherwise default version defined in main `CMakeLists.txt` will be applied
````bash
export GNA_VERSION=<number of specific gna version>
cmake -S $GNA_PLUGIN_SOURCE_DIR -B $GNA_PLUGIN_BUILD_DIR -D OpenVINODeveloperPackage_DIR=$DEVELOPER_PACKAGE_DIR -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON -D GNA_LIB_VERSION=$GNA_VERSION
````

Run build
````bash
cmake --build $GNA_PLUGIN_BUILD_DIR --config $BUILD_TYPE
````
> **Note**
> Developer package has bo be built for respective configuration type otherwise gna plugin build will fail.



Build and run plugin's tests
----------------------------

##### Using ctest

By default all tests are compiled during plugin build, however there is a possibility to selectively compile selected tests by using `--target` option.

For example to compile unit tests use `ov_gna_unit_tests` target run
````bash
cmake --build $GNA_PLUGIN_BUILD_DIR --config $BUILD_TYPE --target ov_gna_unit_tests
````

and simply run
````bash
cd $GNA_PLUGIN_BUILD_DIR
ctest -C $BUILD_TYPE
````

In order to get detailed output use `-V` parameter of `ctest`
````bash
ctest -C $BUILD_TYPE -V
````

##### Manually

There is a possibility to run test from selected target manually. All test binaries reside in `$GNA_PLUGIN_BUILD_DIR/bin/intel64/$BUILD_TYPE`

In order to run test target manually there is a need to point where necessary shared libraries are located
````bash
export TBB_BIN_PATH=$OPENVINO_SOURCE_DIR/temp/tbb/bin
export OV_BIN_PATH=$OPENVINO_SOURCE_DIR/bin/intel64/$BUILD_TYPE
export GNA_LIB_VERSION=<set specific GNA library version>
export GNALIB_BIN_PATH=$OPENVINO_SOURCE_DIR/temp/gna_$GNA_LIB_VERSION/win64/x64
````

In the same terminal run respective test binary

 - legacy transformation tests
````bash
$GNA_PLUGIN_BUILD_DIR/bin/intel64/$BUILD_TYPE/ov_legacy_transformations_tests
````
 - GNA plugin unit tests
````bash
$GNA_PLUGIN_BUILD_DIR/bin/intel64/$BUILD_TYPE/ov_gna_unit_tests
````
 - GNA plugin functional tests
````bash
$GNA_PLUGIN_BUILD_DIR/bin/intel64/$BUILD_TYPE/ov_gna_func_tests --gtest_filter=*smoke*-*DeviceNoThrow*:*HeteroNoThrow*
````



Handy tips
----------

Sometimes it is handy to run cmake project generation in trace mode in order to figure out what happened. In order to do this add cmake's `--trace-source=<name of cmakefile to trace>` option. Besides option `--trace-expand` will expand variable into their values.

For example command below will print out trace messsages from `libGNAConfig.cmake` file
````bash
cmake -S $GNA_PLUGIN_SOURCE_DIR -B $GNA_PLUGIN_BUILD_DIR -D OpenVINODeveloperPackage_DIR=$DEVELOPER_PACKAGE_DIR -D OV_ROOT_DIR=$OPENVINO_SOURCE_DIR --trace-source=libGNAConfig.cmake --trace-expand
````



References
----------

https://docs.openvino.ai/2022.1/openvino_docs_ie_plugin_dg_plugin_build.html
https://docs.openvino.ai/2022.3/openvino_docs_ie_plugin_dg_plugin_build.html
