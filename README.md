## Plugin Compilation


#### Assumptions

- OpenVINO version used 2023.0 (**branch releases/2023/0**)
- Commands syntax is bash-based
- There are following (environmental) variables defined:
````bash
export openvino_source_dir = <path to local openvino source directory>
export developer_package_dir = <path to generated developer package directory>
export build_type = <Release|Debug|etc.>
export gna_plugin_source_dir = <path to local gna plugin souce directory>
export gna_plugin_build_dir = <path to generated gna plugin build>
````


#### Prerequisties

Switch to respective version of openvino
````bash
cd $openvino_source_dir
git checkout releases/2023/0
````

Prepare openvino developer package (build openvino)
````bash
cmake -S $openvino_source_dir -B $developer_package_dir -D CMAKE_BUILD_TYPE=$build_type -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON
cmake --build $developer_package_dir --target ov_dev_targets --config $build_type
````
> **Note**
> - It's important to keep in mind that `CMAKE_BUILD_TYPE` has no effect for multi-configuration generators such as Visual Studio. Therefore in this case config type has to be provided to build command as above.
> - There is yet another target `ie_dev_targets` can be used, but it seems to have identical result as `ov_dev_targets`

Clone **gna.plugin** repository
````bash
git clone https://github.com/intel-innersource/frameworks.ai.gna.plugin.git $gna_plugin_source_dir
````


#### Plugin build

Create **gna.plugin** solution
````bash
cmake -S $gna_plugin_source_dir -B $gna_plugin_build_dir -D OpenVINODeveloperPackage_DIR=$developer_package_dir -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON
````

It happens often that for development purposes many versions of gna library exist in openvino `temp\` directory. Therefore it is possible to use specific version of the library by defining `GNA_LIB_VERSION` otherwise default version defined in main `CMakeLists.txt` will be applied
````bash
export gna_version=<number of specific gna version>
cmake -S $gna_plugin_source_dir -B $gna_plugin_build_dir -D OpenVINODeveloperPackage_DIR=$developer_package_dir -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON -D GNA_LIB_VERSION=$gna_version
````


Run build
````bash
cmake --build $gna_plugin_build_dir --config $build_type
````
> **Note**
> Developer package has bo be built for respective configuration type otherwise gna plugin build will fail.

#### Plugin's tests build and run

By default all tests are compiled during plugin build, however there is a possibility to selectively compile selected tests by using `--target` option.

For example to compile unit tests use `ov_gna_unit_tests` target run
````bash
cmake --build $gna_plugin_build_dir --config $build_type --target ov_gna_unit_tests
````

In order to run tests do
````bash
cd $gna_plugin_build_dir
ctest -C $build_type
````

> **Note**
> Currently only `ov_gna_unit_tests` are supported

#### Handy tips

Sometimes it is handy to run cmake project generation in trace mode in order to figure out what happened. In order to do this add cmake's `--trace-source=<name of cmakefile to trace>` option. Besides option `--trace-expand` will expand variable into their values.

For example command below will print out trace messsages from `libGNAConfig.cmake` file
````bash
cmake -S $gna_plugin_source_dir -B $gna_plugin_build_dir -D OpenVINODeveloperPackage_DIR=$developer_package_dir -D OV_ROOT_DIR=$openvino_source_dir --trace-source=libGNAConfig.cmake --trace-expand
````

#### References

https://docs.openvino.ai/2022.1/openvino_docs_ie_plugin_dg_plugin_build.html
https://docs.openvino.ai/2022.3/openvino_docs_ie_plugin_dg_plugin_build.html
