//SEMAFORI NOTTE

#include"functions.h"

// Funzione per ottenere i nomi dei file in una directory usando OpenCV
std::vector<std::string> getVideoFiles(const std::string& folderPath)
{
    std::vector<std::string> videoPaths;
    std::vector<cv::String> files;
    cv::glob(folderPath, files);

    for (const auto& file : files)
    {
        if (file.find(".mp4") != std::string::npos)
        {
            videoPaths.push_back(file);
        }
    }
    return videoPaths;
}

int main()
{
    try
    {
        std::string folderPath = std::string(EXAMPLE_IMAGES_PATH) + "/semafori_notte/*";
        std::vector<std::string> videoPaths = getVideoFiles(folderPath);

        for (int i = 0; i < videoPaths.size(); i++)
        {
            ipa::processVideoStream(videoPaths[i], TLR::frameProcessor, "", true, 0, 0);
        }

        return EXIT_SUCCESS;
    }
    catch (ipa::error& ex)
    {
        std::cout << "EXCEPTION thrown by " << ex.getSource() << " source :\n\t|=> " << ex.what() << std::endl;
    }
    catch (ucas::Error& ex)
    {
        std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
    }
    catch (std::exception& ex)
    {
        std::cout << "Standard exception: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught!" << std::endl;
    }

    return EXIT_FAILURE;
}

