$openvino_version = Get-Content .\openvino_version.txt -RAW;
$openvino_src_dir="C:\openvino_$openvino_version";
$developer_package_dir="$openvino_src_dir\bin";
$gna_plugin_src_dir='.';

New-Item -ItemType Directory -Path $env:BUILD_DIR
cmake -S $gna_plugin_src_dir -B $env:BUILD_DIR -D OpenVINODeveloperPackage_DIR=$developer_package_dir -D ENABLE_TESTS=ON -D ENABLE_FUNCTIONAL_TESTS=ON
cmake --build $env:BUILD_DIR --config $env:BUILD_TYPE
