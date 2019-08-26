add_library( mylib SHARED
    cyl_projection.cpp
    cyl_stitch.cpp
    cyl_fundam.cpp
    cyl_gcSeamFinder.cpp
)

target_link_libraries( mylib
    ${THIRD_PARTY_LIBS}
)
