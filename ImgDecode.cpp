// This file implements decoding for the logical code frame.
#include "ImgDecode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "code.h"
#include "pic.h"

namespace ImageDecode
{
	constexpr int MaxDataLength = 10000;

	enum color
	{
		Black = 0,
		Blue = 1,
		Green = 2,
		Cyan = 3,
		Red = 4,
		Magenta = 5,
		Yellow = 6,
		White = 7
	};

	struct DataArea
	{
		int top;
		int left;
		int height;
		int width;
		int trimRight;
	};

	struct CellPos
	{
		int row;
		int col;
	};

	enum class FrameType
	{
		Start = 0,
		End = 1,
		StartAndEnd = 2,
		Normal = 3
	};

	constexpr int SmallQrPointRadius = 3;
	constexpr int CornerReserveSize = 21;
	constexpr int HeaderLeft = 21;
	constexpr int HeaderTop = 3;
	constexpr int HeaderFieldBits = 16;
	constexpr int HeaderWidth = 16;
	constexpr int TopDataLeft = HeaderLeft + HeaderWidth;
	constexpr int TopDataWidth = 75;
	constexpr int DataAreaCount = 5;
	constexpr int PaddingCellCount = 4;
	constexpr int BitsPerCell = 3;

	const std::array<DataArea, DataAreaCount> kDataAreas =
	{{
		{3, TopDataLeft, 3, TopDataWidth, 0},
		{6, 21, 15, 91, 0},
		{21, 3, 88, 127, 0},
		{109, 3, 3, 127, 0},
		{112, 21, 18, 91, 0}
	}};


static const cv::Vec3b g_standardColors[8] = {
    cv::Vec3b(0, 0, 0),       // 黑色: 000
    cv::Vec3b(0, 0, 255),     // 红色: 001
    cv::Vec3b(0, 255, 0),     // 绿色: 010
    cv::Vec3b(0, 255, 255),   // 黄色: 011
    cv::Vec3b(255, 0, 0),     // 蓝色: 100
    cv::Vec3b(255, 0, 255),   // 品红: 101
    cv::Vec3b(255, 255, 0),   // 青色: 110
    cv::Vec3b(255, 255, 255)  // 白色: 111
};

static int g_colorThreshold = 128;
static bool g_colorCalibrated = false;

float colorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) {
    int db = color1[0] - color2[0];
    int dg = color1[1] - color2[1];
    int dr = color1[2] - color2[2];
    return std::sqrt(db*db + dg*dg + dr*dr);
}

int findClosestStandardColor(const cv::Vec3b& pixel) {
    int closestIndex = 0;
    float minDistance = std::numeric_limits<float>::max();
    
    for (int i = 0; i < 8; i++) {
        float dist = colorDistance(pixel, g_standardColors[i]);
        if (dist < minDistance) {
            minDistance = dist;
            closestIndex = i;
        }
    }
    
    return closestIndex;
}

void calibrateColors(cv::Mat& mat) {
    if (g_colorCalibrated) return;
    
    std::vector<cv::Vec3b> standardColors = {
        cv::Vec3b(0, 0, 0),
        cv::Vec3b(255, 0, 0),
        cv::Vec3b(0, 255, 0),
        cv::Vec3b(0, 255, 255),
        cv::Vec3b(0, 0, 255),
        cv::Vec3b(255, 0, 255),
        cv::Vec3b(255, 255, 0),  
        cv::Vec3b(255, 255, 255)  
    };
    
    // 网格大小：8x8像素
	//
    const int gridSize = 8;
    const int gridRows = mat.rows / gridSize;
    const int gridCols = mat.cols / gridSize;
    
    // 对每个网格进行统一颜色校准
	//
    for (int gridY = 0; gridY < gridRows; gridY++) {
        for (int gridX = 0; gridX < gridCols; gridX++) {
            // 计算网格的平均颜色
			//
            int startY = gridY * gridSize;
            int startX = gridX * gridSize;
            
            long totalR = 0, totalG = 0, totalB = 0;
            int pixelCount = 0;
            
            for (int y = startY; y < startY + gridSize && y < mat.rows; y++) {
                for (int x = startX; x < startX + gridSize && x < mat.cols; x++) {
                    cv::Vec3b pixel = mat.at<cv::Vec3b>(y, x);
                    totalB += pixel[0];
                    totalG += pixel[1];
                    totalR += pixel[2];
                    pixelCount++;
                }
            }
            
            if (pixelCount == 0) continue;
            
            // 计算平均颜色
			//
            cv::Vec3b avgColor(
                totalB / pixelCount,
                totalG / pixelCount,
                totalR / pixelCount
            );
            
            // 以128为阈值，找到最接近的标准颜色
			//
            int minDistance = INT_MAX;
            cv::Vec3b closestColor = avgColor;
            
            for (const auto& stdColor : standardColors) {
                int distance = std::abs(avgColor[0] - stdColor[0]) + 
                              std::abs(avgColor[1] - stdColor[1]) + 
                              std::abs(avgColor[2] - stdColor[2]);
                
                if (distance < minDistance) {
                    minDistance = distance;
                    closestColor = stdColor;
                }
            }
            
            // 将整个网格统一设置为该标准颜色
			//
            for (int y = startY; y < startY + gridSize && y < mat.rows; y++) {
                for (int x = startX; x < startX + gridSize && x < mat.cols; x++) {
                    mat.at<cv::Vec3b>(y, x) = closestColor;
                }
            }
        }
    }
    
    std::cout << "Grid-based color calibration completed - each 8x8 grid has uniform color" << std::endl;
    g_colorCalibrated = true;
}

bool isWhiteCell(const Vec3b& cell)
{
	return cell[0] + cell[1] + cell[2] >= 384;
}

int readCellValue(const Vec3b& cell)
{
	
	const int b = cell[0];
	const int g = cell[1];
	const int r = cell[2];
	
	
	int b_binary = (b >= 128) ? 255 : 0;
	int g_binary = (g >= 128) ? 255 : 0;
	int r_binary = (r >= 128) ? 255 : 0;
	
	
	cv::Vec3b binaryColor(b_binary, g_binary, r_binary);
	
	
	int closestColorIndex = findClosestStandardColor(binaryColor);
	
	return closestColorIndex;
}

int readCellValueFromGrid(const Mat& mat, int gridRow, int gridCol)
{
	const int gridSize = 8;
	const int numGrids = 133;
	
	int centerX = gridCol * gridSize + gridSize / 2;
	int centerY = gridRow * gridSize + gridSize / 2;
	
	if (centerX >= mat.cols) centerX = mat.cols - 1;
	if (centerY >= mat.rows) centerY = mat.rows - 1;
	
	cv::Vec3b pixel = mat.at<cv::Vec3b>(centerY, centerX);
	
	return readCellValue(pixel);
}

	bool isInsideCornerQuietZone(int row, int col)
	{
		return row >= 130 || col >= 130;
	}

	bool isInsideCornerSafetyZone(int row, int col)
	{
		const int center = ImageDecode::FrameSize - ImageDecode::SmallQrPointbias;
		return std::abs(row - center) <= ImageDecode::SmallQrPointRadius + 2 && std::abs(col - center) <= ImageDecode::SmallQrPointRadius + 2;
	}

	std::vector<CellPos> buildAreaCells(const DataArea& area)
	{
		std::vector<CellPos> cells;
		for (int row = area.top; row < area.top + area.height; ++row)
		{
			const int rowWidth = area.width - area.trimRight;
			for (int col = area.left; col < area.left + rowWidth; ++col)
			{
				cells.push_back({row, col });
			}
		}
		return cells;
	}

	std::vector<CellPos> buildCornerDataCells()
	{
		std::vector<CellPos> cells;
		for (int row = ImageDecode::FrameSize - ImageDecode::CornerReserveSize; row < ImageDecode::FrameSize; ++row)
		{
			for (int col = ImageDecode::FrameSize - ImageDecode::CornerReserveSize; col < ImageDecode::FrameSize; ++col)
			{
				if (isInsideCornerQuietZone(row, col))
				{
					continue;
				}
				if (isInsideCornerSafetyZone(row, col))
				{
					continue;
				}
				cells.push_back({ row, col });
			}
		}
		return cells;
	}

	std::vector<CellPos> buildMergedDataCells()
	{
		std::vector<CellPos> cells;
		for (const auto& area : kDataAreas)
		{
			const auto areaCells = buildAreaCells(area);
			cells.insert(cells.end(), areaCells.begin(), areaCells.end());
		}
		const auto cornerCells = buildCornerDataCells();
		cells.insert(cells.end(), cornerCells.begin(), cornerCells.end());
		if (cells.size() > PaddingCellCount)
		{
			cells.resize(cells.size() - PaddingCellCount);
		}
		return cells;
	}

	std::vector<CellPos> getPaddingCells()
	{
		std::vector<CellPos> cells;
		for (const auto& area : kDataAreas)
		{
			const auto areaCells = buildAreaCells(area);
			cells.insert(cells.end(), areaCells.begin(), areaCells.end());
		}
		const auto cornerCells = buildCornerDataCells();
		cells.insert(cells.end(), cornerCells.begin(), cornerCells.end());
		if (cells.size() <= PaddingCellCount)
		{
			return {};
		}
		return std::vector<CellPos>(cells.end() - PaddingCellCount, cells.end());
	}

	int readTailLenHighBit(const Mat& mat)
	{
		int whiteCount = 0;
		for (const auto& cell : getPaddingCells())
		{
			if (isWhiteCell(mat.at<Vec3b>(cell.row, cell.col)))
			{
				++whiteCount;
			}
		}
		return whiteCount * 2 >= PaddingCellCount ? 1 : 0;
	}

	uint16_t readHeaderField(const Mat& mat, int fieldId)
{
	uint16_t value = 0;
	const int row = HeaderTop + fieldId;
	
	if (mat.rows == 1064 && mat.cols == 1064) {
		for (int bit = 0; bit < HeaderFieldBits; ++bit)
		{
			int gridCol = HeaderLeft + bit;
			int gridRow = row;
			
			if (gridRow >= 0 && gridRow < 133 && gridCol >= 0 && gridCol < 133) {
				int cellValue = readCellValueFromGrid(mat, gridRow, gridCol);
				
				if (cellValue == White) {
					value |= static_cast<uint16_t>(1u << bit);
				}
			}
		}
	} else {
		for (int bit = 0; bit < HeaderFieldBits; ++bit)
		{
			if (isWhiteCell(mat.at<Vec3b>(row, HeaderLeft + bit)))
			{
				value |= static_cast<uint16_t>(1u << bit);
			}
		}
	}
	
	return value;
}

	FrameType parseFrameType(uint16_t headerValue, bool& isStart, bool& isEnd)
	{
		const uint16_t flagBits = headerValue & 0xF;
		switch (flagBits)
		{
		case 0b0011:
			isStart = true;
			isEnd = false;
			return FrameType::Start;
		case 0b1100:
			isStart = false;
			isEnd = true;
			return FrameType::End;
		case 0b1111:
			isStart = true;
			isEnd = true;
			return FrameType::StartAndEnd;
		default:
			isStart = false;
			isEnd = false;
			return FrameType::Normal;
		}
	}

	void readPayload(const Mat& mat, std::vector<unsigned char>& info)
{
	const auto cells = buildMergedDataCells();
	info.assign(ImageDecode::BytesPerFrame, 0);
	const int totalCells = (ImageDecode::BytesPerFrame * 8 + BitsPerCell - 1) / BitsPerCell;
	const int mask = (1 << BitsPerCell) - 1;
	
	for (int cellIndex = 0;
		cellIndex < totalCells && cellIndex < static_cast<int>(cells.size());
		++cellIndex)
	{
		const int bitIndex = cellIndex * BitsPerCell;
		const int byteIndex = bitIndex / 8;
		const int offset = bitIndex % 8;
		
		int value;
		if (mat.rows == 1064 && mat.cols == 1064) {
			int gridRow = cells[cellIndex].row;
			int gridCol = cells[cellIndex].col;
			value = readCellValueFromGrid(mat, gridRow, gridCol);
		} else {
			value = readCellValue(mat.at<Vec3b>(cells[cellIndex].row, cells[cellIndex].col));
		}
		
		info[byteIndex] |= static_cast<unsigned char>((value & mask) << offset);
		if (offset + BitsPerCell > 8 && byteIndex + 1 < ImageDecode::BytesPerFrame)
		{
			info[byteIndex + 1] |= static_cast<unsigned char>(value >> (8 - offset));
		}
	}
}

	bool hasLegalSize(const Mat& mat)
{
	return (mat.rows == 1064 && mat.cols == 1064 && mat.type() == CV_8UC3) ||
	       (mat.rows == ImageDecode::FrameSize && mat.cols == ImageDecode::FrameSize && mat.type() == CV_8UC3);
}

	bool ImageDecode::Main(Mat& mat, ImageInfo& imageInfo)
{
	imageInfo.Info.clear();
	imageInfo.CheckCode = 0;
	imageInfo.FrameBase = 0;
	imageInfo.IsStart = false;
	imageInfo.IsEnd = false;

	if (!hasLegalSize(mat)) {
		std::cout << "Image size error: " << mat.rows << "x" << mat.cols << ", type: " << mat.type() << std::endl;
		return false;
	}
	
	Mat processedMat;
	if (!ImgParse::Main(mat, processedMat)) {
		std::cout << "QR code processing failed" << std::endl;
		return false;
	}

	calibrateColors(processedMat);

	std::cout << "DEBUG: Image size: " << processedMat.cols << "x" << processedMat.rows << std::endl;
	std::cout << "DEBUG: Before calling readHeaderField" << std::endl;
	std::cout.flush();

	const uint16_t headerValue = readHeaderField(processedMat, 0);
	std::cout << "Header value: " << headerValue << std::endl;
	
	parseFrameType(headerValue, imageInfo.IsStart, imageInfo.IsEnd);
	std::cout << "Frame type - Start: " << imageInfo.IsStart << ", End: " << imageInfo.IsEnd << std::endl;
	
	const int codeLenLow = headerValue >> 4;
	const int codeLen = imageInfo.IsEnd
		? (codeLenLow | (readTailLenHighBit(processedMat) << 12))
		: ImageDecode::BytesPerFrame;
	
	std::cout << "Calculated data length: " << codeLen << std::endl;
	
	if (codeLen > ImageDecode::MaxDataLength)
	{
		std::cout << "Data length exceeds limit, skipping frame" << std::endl;
		return true;
	}

		imageInfo.CheckCode = readHeaderField(processedMat, 1);
		imageInfo.FrameBase = readHeaderField(processedMat, 2);

		std::vector<unsigned char> payload;
		readPayload(processedMat, payload);
		payload.resize(codeLen);
		imageInfo.Info.swap(payload);

		return imageInfo.CheckCode != Code::CalCheckCode(
			imageInfo.Info.data(),
			codeLen,
			imageInfo.IsStart,
			imageInfo.IsEnd,
			imageInfo.FrameBase
		);
	}
}
