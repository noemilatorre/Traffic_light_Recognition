#include "functions.h"

namespace TLR
{
	// Parameters
	double min_width_perc = 0.005;
	double max_width_perc = 0.05;
	double min_circularity = 0.65;
	double min_area = 100;
	double max_area = 2500;
	double max_aspect_ratio = 1.6;
	double min_aspect_ratio = 0.7;
	int min_saturation = 100;
	int min_illumination = 252;

	int red_reference = 0;				// tinta di riferimento per la classe "rosso" 
	int green_reference = 110;			// tinta di riferimento per la classe "verde" 
	int orange_reference = 30;			// tinta di riferimento per la classe "arancione"	

	int red_hue_min1 = 0;
	int red_hue_max1 = 20;
	int red_hue_min2 = 340;
	int red_hue_max2 = 360;
	int yellow_hue_min = 20;
	int yellow_hue_max = 80;
	int green_hue_min = 100;
	int green_hue_max = 180;

	int majority_voting_window = 35;

	std::list < Detection > detections;

	//costruttore 
	Detection::Detection(Color c, int f, double g) : color(c), frame(f), goodness(g) {}

	double rad2deg(double radians)
	{
		return radians * (180.0 / ucas::PI);
	}

	double deg2rad(double degrees)
	{
		return degrees * (ucas::PI / 180.0);
	}

	cv::Scalar color2scalar(Color color)
	{
		if (color == Color::GREEN)
			return cv::Scalar(0, 255, 0);
		else if (color == Color::ORANGE)
			return cv::Scalar(0, 215, 255);
		else if (color == Color::RED)
			return cv::Scalar(0, 0, 255);
		else
			return cv::Scalar(128, 128, 128);
	}

	std::string color2text(Color color)
	{
		if (color == Color::GREEN)
			return "G0";
		else if (color == Color::ORANGE)
			return "SLOW DOWN";
		else if (color == Color::RED)
			return "STOP";
		else
			return "...";
	}

	double avgValInContour(
		const cv::Mat& img,
		const std::vector<cv::Point>& object,
		bool cosine_correction = false)
	{
		double sum = 0;
		int count = 0;
		cv::Rect bbox = cv::boundingRect(object);
		for (int y = bbox.y; y < bbox.y + bbox.height; y++)
		{
			const unsigned char* yRow = img.ptr<unsigned char>(y);
			for (int x = bbox.x; x < bbox.x + bbox.width; x++)
				//restituisce valore positivo se il punto è all'interno del contorno
				if (cv::pointPolygonTest(object, cv::Point(x, y), false) > 0) 
				{
					if (cosine_correction)
						sum += rad2deg(std::acos(std::cos(deg2rad(yRow[x] * 2.0)))) / 2.0;
					else
						sum += yRow[x];
					count++;
				}
		}

		return sum / count;
	}

	double goodness(double C, double avgV, double avgS, double areaN)
	{
		return (1.5 * C + (avgV / 255.0) + (avgS / 255.0) + areaN) / 4;
	}

	double distanceFromColors(double avg_hue)
	{
		double dist_to_orange = std::abs(orange_reference - avg_hue);
		double dist_to_green = std::abs(green_reference - avg_hue);
		double dist_to_red = std::abs(red_reference - avg_hue);

		return std::min({ dist_to_orange, dist_to_green, dist_to_red });
	}

	bool isValidBoundingBox(const cv::Rect& brect, int min_width, int max_width)
	{
		return (brect.width < min_width || brect.width > max_width);
	}

	bool isValidAreaRatio(double area, double area_brect) {
		double ratio = area / area_brect;
		return ratio < 0.55 || ratio > 0.85;
	}

	bool isValidArea(double area) {
		return area < min_area || area > max_area;
	}

	bool isValidCircularity(double circularity) {
		return circularity < min_circularity;
	}

	bool isValidAspectRatio(double aspect_ratio) {
		return aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio;
	}

	bool isValidHue(double avg_hue) {
		return (avg_hue < red_hue_min1 || avg_hue > red_hue_max1) &&
			(avg_hue < red_hue_min2 || avg_hue > red_hue_max2) &&
			(avg_hue < yellow_hue_min || avg_hue > yellow_hue_max) &&
			(avg_hue < green_hue_min || avg_hue > green_hue_max);
	}

	bool isValidIllumination(double avg_illumination) {
		return avg_illumination < min_illumination;
	}

	bool isValidSaturation(double avg_saturation) {
		return avg_saturation < min_saturation;
	}

	void gammaTransformation(const cv::Mat& frame, cv::Mat& frame_out, float gamma)
	{
		int L = 256;
		float c = std::pow(L - 1, 1 - gamma);

		for (int y = 0; y < frame_out.rows; y++)
		{
			unsigned char* yRow = frame_out.ptr<unsigned char>(y);
			for (int x = 0; x < frame_out.cols; x++)
				yRow[x] = c * std::pow(yRow[x], gamma) + 0.5f;
		}
	}

	void preProcessing(const cv::Mat& frame, std::vector<cv::Mat>& frame_hsv_chans) {

		cv::Mat frame_hsv;
		cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
		cv::split(frame_hsv, frame_hsv_chans);

		// Applico la trasformazione gamma sul canale di saturazione (S)
		gammaTransformation(frame_hsv_chans[1], frame_hsv_chans[1], 0.2);

		// Aumento la luminosità del canale di valore (V)
		double alpha = 2; // Fattore di moltiplicazione per aumentare la luminosità
		frame_hsv_chans[2].convertTo(frame_hsv_chans[2], -1, alpha, 0);
	}

	void binarizeAndMorph(const cv::Mat& input_channel, cv::Mat& output_image) {
		cv::Mat frame_binarized;
		cv::threshold(input_channel, frame_binarized, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);

		cv::morphologyEx(frame_binarized, output_image, cv::MORPH_ERODE,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
		cv::morphologyEx(output_image, output_image, cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	
	}

	void detection(std::vector <std::vector<cv::Point>> objects,
		int min_width, int max_width, int& number_object,
		std::vector<std::vector<cv::Point>>& frame_detections_objects,
		std::vector<cv::Mat>& frame_hsv_chans,
		std::vector<double>& score, std::vector<double>& g)
	{
		for (size_t k = 0; k < objects.size(); k++) {

			cv::Rect brect = cv::boundingRect(objects[k]);

			if (isValidBoundingBox(brect, min_width, max_width)) continue;

			// Verifiche validità dell'oggetto
			double area_brect = brect.width * brect.height;
			double area = cv::contourArea(objects[k]);
			double perim = cv::arcLength(objects[k], true);
			double circularity = 4 * ipa::PI * area / (perim * perim);
			double aspect_ratio = static_cast<double>(brect.width) / brect.height;		

			if (isValidAreaRatio(area, area_brect)) continue;
			if (isValidArea(area)) continue;
			if (isValidCircularity(circularity)) continue;
			if (isValidAspectRatio(aspect_ratio)) continue;

			// Calcolo tinta media dell'oggetto
			double avg_hue = 2 * avgValInContour(frame_hsv_chans[0], objects[k], true);
			if (isValidHue(avg_hue)) continue;

			// Analizzo i pixel se si trovano nel range di tinta di verde, giallo/arancione o rosso 
			// assegno ai pixel il valore di verde,giallo/arancione o rosso di riferimento
			cv::Rect bbox = cv::boundingRect(objects[k]);
			for (int y = bbox.y; y < bbox.y + bbox.height; y++)
			{
				unsigned char* yRow = frame_hsv_chans[0].ptr<unsigned char>(y);
				for (int x = bbox.x; x < bbox.x + bbox.width; x++)
					if (cv::pointPolygonTest(objects[k], cv::Point(x, y), false) > 0)
					{
						if ((avg_hue > 150 && avg_hue < 180))
							yRow[x] = green_reference;
						else if ((avg_hue > 20 && avg_hue < 35) || (avg_hue > 50 && avg_hue < 60))
							yRow[x] = orange_reference;
						else if ((avg_hue > 0 && avg_hue < 20) || (avg_hue > 340 && avg_hue <= 360))
							yRow[x] = red_reference;
					}
			}

			// Calcolo tinta media dell'oggetto di nuovo perchè sono cambiati i valori di tinta di qualche pxel
			avg_hue = 2 * avgValInContour(frame_hsv_chans[0], objects[k], true);

			// Verifico illuminazione media dell'oggetto
			double avg_illumination = avgValInContour(frame_hsv_chans[2], objects[k]);
			if (isValidIllumination(avg_illumination)) continue;

			// Verifico saturazione media dell'oggetto
			double avg_saturation = avgValInContour(frame_hsv_chans[1], objects[k]);
			if (isValidSaturation(avg_saturation)) continue;

			// Aggiungo l'oggetto rilevato
			frame_detections_objects.push_back(objects[k]);

			g[number_object] = goodness(circularity, avg_illumination, avg_saturation, ((area - min_area) / (max_area - min_area)));
			score[number_object] = g[number_object] - (distanceFromColors(avg_hue) / 360); //quanto è buono il risultato //divido per 360 per normalizzare tra 0 e 1
			number_object++;
		}
	}

	void recognition(const std::vector<std::vector<cv::Point>>& frame_detections_objects,
		std::vector<double>& score,
		std::vector<double>& g,
		int number_object,
		const std::vector<cv::Mat>& frame_hsv_chans, int frame_count,
		std::list<Detection>& frame_detections, cv::Mat& frame_out)
	{
		// Ordina i valori di bontà dal più grande
		std::sort(score.begin(), score.end(), std::greater<double>());

		// Considero solo i 2 oggetti con il punteggio maggiore
		if (frame_detections_objects.size() >= 2)
		{
			// Trovo gli object con bontà/validità più alti, gli assegno il colore disegno i contorni
			for (int i = 0; i < 2; i++)
			{
				if (score[i] > 0.7)
				{
					double bonta = 0;
					int indice = 0;
					Color light_color;

					for (int j = 0; j < frame_detections_objects.size(); j++)
					{
						double avg_hue = 2 * avgValInContour(frame_hsv_chans[0], frame_detections_objects[j], true);

						// Assegno colore a ogni oggetto
						if ((avg_hue >= red_hue_min1 && avg_hue <= red_hue_max1) ||
							(avg_hue >= red_hue_min2 && avg_hue <= red_hue_max2))
							light_color = Color::RED;
						else if ((avg_hue >= yellow_hue_min && avg_hue <= yellow_hue_max))
							light_color = Color::ORANGE;
						else if ((avg_hue >= green_hue_min && avg_hue <= green_hue_max))
							light_color = Color::GREEN;

						bonta = g[j] - (distanceFromColors(avg_hue) / 360);

						// Trovo l'indice dell'oggetto corrente con bontà/validità maggiore
						if (bonta == score[i])
						{
							indice = j;
							break;
						}
					}
					frame_detections.push_back(Detection(light_color, frame_count, bonta));
					cv::drawContours(frame_out, frame_detections_objects, indice, color2scalar(light_color), 7, cv::LINE_AA);
				}
			}
		}
	}

	void majorityVoting(
		int frame_count,
		std::list<Detection>& frame_detections,
		cv::Mat& frame_out)
	{
		// Colore del frame corrente = weighted (by goodness) voting
		std::map<Color, double> weighted_voting;
		std::map<Color, int> number_element;

		// Per ogni oggetto investigato nel frame corrente
		for (auto detection : frame_detections)
		{
			std::string colorName = color2text(detection.color);
			weighted_voting[detection.color] += detection.goodness; // Sommo la bontà per ogni colore
			number_element[detection.color] += 1; // Conto elementi rilevati per ogni colore
		}

		// Determina il colore dominante per il frame corrente basato sulla bontà di ciascun colore
		double max_accumulated_goodness = 0;
		Color frame_color = Color::UNKNOWN;
		for (auto it : weighted_voting)
			if ((it.second / number_element[it.first]) >= max_accumulated_goodness)
			{
				max_accumulated_goodness = it.second;
				frame_color = it.first;
			}
		if (frame_detections.size())
		{
			detections.push_back(Detection(frame_color, frame_count, max_accumulated_goodness));
		}

		// Moving window
		while (detections.back().frame - detections.front().frame >= majority_voting_window)
			detections.pop_front();

		// Majority voting nel tempo
		std::map<Color, int> color_counts;
		for (auto& detection : detections)
			color_counts[detection.color]++;

		// Trova il colore più votato
		int max_count = 0;
		Color most_voted_color = Color::UNKNOWN;
		for (auto& it : color_counts)
		{
			if (it.second > max_count)
			{
				max_count = it.second;
				most_voted_color = it.first;
			}
		}

		cv::Scalar cv_color = color2scalar(most_voted_color);
		cv::putText(frame_out, color2text(most_voted_color), cv::Point(100, 100), 2, 4, color2scalar(most_voted_color), 3);
	}


	cv::Mat frameProcessor(const cv::Mat& frame)
	{
		cv::Mat frame_out = frame.clone();

		// Variabile che conta i frame e serve per il majoring voting
		static int frame_count = 0;
		frame_count++;

		// Parametri del rettangolo che conterrà gli oggetti
		int min_width = ucas::round<double>(min_width_perc * frame.cols);
		int max_width = ucas::round<double>(max_width_perc * frame.cols);

		// Array per contenere i canali HSV
		std::vector<cv::Mat> frame_hsv_chans;
		preProcessing(frame, frame_hsv_chans);

		// Binarizzo e applico operazioni morfologiche 
		cv::Mat frame_binarized_out;
		binarizeAndMorph(frame_hsv_chans[2], frame_binarized_out);

		// Struttura che conterrà gli oggetti rilevati
		std::list < Detection > frame_detections;

		// Estraggo tutti i bordi 
		std::vector < std::vector < cv::Point > > objects;
		std::vector < std::vector < cv::Point > > frame_detections_objects;
		cv::findContours(frame_binarized_out, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		// Variabili che conterranno la bontà/validità di ogni oggetto 
		std::vector<double> score(objects.size());
		std::vector<double> g(objects.size(), 0.0);

		int number_object = 0;

		detection(objects, min_width, max_width, number_object, frame_detections_objects, frame_hsv_chans, score, g);

		recognition(frame_detections_objects, score, g, number_object, frame_hsv_chans, frame_count, frame_detections, frame_out);

		majorityVoting(frame_count, frame_detections, frame_out);

		return frame_out;
	}
}


