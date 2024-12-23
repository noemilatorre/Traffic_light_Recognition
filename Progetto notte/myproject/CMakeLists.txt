# include libraries
include_directories (${ipa_SOURCE_DIR}/3rdparty/ipalib)
include_directories (${ipa_SOURCE_DIR}/3rdparty/ucaslib)

# find sources
file(GLOB src *.h *.hpp *.cpp)

# fix va_start error in VS 14+
add_definitions(-D_CRT_NO_VA_START_VALIDATION)

# create executable from sources
add_executable(ipaproject ${src})

# link the executable to other modules / libraries 
target_link_libraries(ipaproject ipalib ucaslib ${OpenCV_LIBS} ${Boost_LIBRARIES})
