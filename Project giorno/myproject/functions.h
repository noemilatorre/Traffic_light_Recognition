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
	double rad2deg(double radians);
	double deg2rad(double degrees);

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
	bool isValidBoundingBox(const cv::Rect& brect, int min_width, int max_width);
	bool isValidAreaRatio(double area, double area_brect);
	bool isValidArea(double area);
	bool isValidCircularity(double circularity);
	bool isValidAspectRatio(double aspect_ratio);
	bool isValidHue(double avg_hue);
	bool isValidIllumination(double avg_illumination);
	bool isValidSaturation(double avg_saturation);

	// Procedura per filtro gamma
	void gammaTransformation(const cv::Mat& frame, cv::Mat& frame_out, float gamma);

	// Procedura per il preprocessing
	void preProcessing(const cv::Mat& frame, std::vector<cv::Mat>& frame_hsv_chans);

	// Procedura per la binarizzazione e morfologia
	void binarizeAndMorph(const cv::Mat& input_channel, cv::Mat& output_image);

	// Procedura per selezionare gli oggetti
	void detection(std::vector <std::vector<cv::Point>> objects,
		int min_width, int max_width, int& number_object,
		std::vector<std::vector<cv::Point>>& frame_detections_objects,
		std::vector<cv::Mat>& frame_hsv_chans,
		std::vector<double>& score, std::vector<double>& g);

	// Procedura per selezionare gli oggetti con maggior bontà/validità
	void recognition(const std::vector<std::vector<cv::Point>>& frame_detections_objects,
		std::vector<double>& score,
		std::vector<double>& g,
		int number_object,
		const std::vector<cv::Mat>& frame_hsv_chans, int frame_count,
		std::list<Detection>& frame_detections, cv::Mat& frame_out);

	// Procedura per il majority voting 
	void majorityVoting(
		int frame_count,
		std::list<Detection>& frame_detections,
		cv::Mat& frame_out);
}

#endif // FUNCTIONS_H
