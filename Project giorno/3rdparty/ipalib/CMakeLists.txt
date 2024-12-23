# find source files
file(GLOB ipa_src *.h *.cpp *.hpp)

add_definitions(-DOPENCV_FACE_DETECTOR_PATH="${ipa_SOURCE_DIR}/3rdparty/ipalib/haarcascade_frontalface_alt.xml")

# create static library from source files
add_library(ipalib STATIC ${ipa_src})