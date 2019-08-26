add_executable( LFStitch_cyl LFStitch_cyl.cpp )
target_link_libraries( LFStitch_cyl mylib )

add_executable( changename changename.cpp )
target_link_libraries( changename mylib )

add_executable( depth depth.cpp )
target_link_libraries( depth mylib )

add_executable( resize resize.cpp )
target_link_libraries( resize mylib )
