#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "ipaConfig.h" 
#include "ucasConfig.h"
#include <list>

namespace TLR
{
	// Enumerazione per i colori
	enum class Color { UNKNOWN, GREEN, ORANGE, RED };

	// Struttura per la rilevazione
	struct Detection
	{
		Color color;
		int frame;
		double goodness;

		Detection(Color _color, int _frame, double _goodness = 0);
	};

	// Funzione per il processamento frame-by-frame
	cv::Mat frameProcessor(const cv::Mat& frame);

	// Funzioni utility per la conversione tra radianti e gradi e l'inverso
	static double rad2deg(double radians);
	static double deg2rad(double degrees);

	// Funzioni utility per la conversione dei colori in scalari e in testo
	cv::Scalar color2scalar(Color color);
	std::string color2text(Color color);

	// Funzione utility per calcolare la media dell'intensità all'interno di un contorno
	double avgValInContour(const cv::Mat& img, const std::vector<cv::Point>& object, bool cosine_correction);

	// Funzione utility per calcolare il punteggio di bontà della rilevazione
	double goodness(double C, double avgV, double avgS, double areaN);

	// Funzione utility per calcolare la distanza dall'hue medio di un oggetto dai valori di riferimento
	double distanceFromColors(double avg_hue);

	// Funzioni di validazione
	bool isValidAreaRatio(double area, double area_brect);
	bool isValidArea(double area);
	bool isValidCircularity(double circularity);
	bool isValidAspectRatio(double aspect_ratio);
	bool isValidHue(double avg_hue);
	bool isValidIllumination(double avg_illumination);
	bool isValidSaturation(double avg_saturation);

	// Procedura filtro gamma
	void gammaTransformation(const cv::Mat& frame, cv::Mat& frame_out, float gamma);

	// Procedura per il preprocessing
	void preProcessing(const cv::Mat& frame, std::vector<cv::Mat>& frame_hsv_chans);

	// Procedura per la binarizzazione e morfologia
	void binarizeAndMorph(const cv::Mat& input_channel, cv::Mat& output_image);

	// Procedura che si occupa di andare a trovare gli oggetti validi (che hanno caratteristiche simili a quelle del semaforo)
	void detectionAndRecognition(std::vector<double>& score,
		int& frame_count,
		std::list<Detection>& frame_detections,
		std::vector <std::vector<cv::Point>> objects,
		cv::Mat& frame_out,
		std::vector<cv::Mat>& frame_hsv_chans);

	// Procedura per il majority voting
	void majorityVoting(
		std::vector<double>& score,
		int frame_count,
		std::list<Detection>& frame_detections,
		cv::Mat& frame_out);
}

#endif // FUNCTIONS_H
