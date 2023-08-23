Set-Location $env:BUILD_DIR;

$openvino_version = Get-Content ..\openvino_version.txt -RAW;
$openvino_source_dir="C:\openvino_$openvino_version";

$pluginf_text=@"
<ie>
    <plugins>
        <plugin name="GNA" location="$env:BUILD_DIR\bin\intel64\$env:BUILD_TYPE\openvino_intel_gna_plugin.dll">
        </plugin>
    </plugins>
</ie>
"@

$pluginf_text | Out-File -FilePath $openvino_source_dir\bin\intel64\Release\plugins.xml

ctest -C Release -V;