set_project("NanoZhGPT_Tokenizer")
set_version("0.1.0")

add_rules("mode.debug", "mode.release")
set_languages("c++17")

add_requires("zlib")

target("byte_bpe_train")
    set_kind("binary")
    add_files("byte_bpe_train.cpp", "byte_bpe_lib.cpp", "byte_bpe_bpe.cpp")
    set_rundir("$(projectdir)")
    add_packages("zlib")
    