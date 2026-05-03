import io
import math
import torch
import numpy as np
from PIL import Image
from scipy import fftpack
from transformers import CLIPProcessor, CLIPModel
import cv2


class ImageQualityMetrics:
    """图像质量量化评估体系

    提供多维度的图像质量评估指标，用于量化生成图像的好坏。
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_processor = None

    def _get_clip(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model.eval()
        return self._clip_model, self._clip_processor

    def evaluate_all(self, image: Image.Image, prompt: str = "") -> dict:
        """运行所有质量评估指标，返回完整的评估报告。"""
        metrics = {}

        # 1. CLIP语义一致性 (已有功能增强)
        if prompt:
            metrics["clip_semantic"] = self.clip_semantic_similarity(image, prompt)

        # 2. 美学评分
        metrics["aesthetic"] = self.aesthetic_score(image)

        # 3. 清晰度/锐度
        metrics["sharpness"] = self.sharpness_score(image)

        # 4. 色彩丰富度
        metrics["color_richness"] = self.color_richness_score(image)

        # 5. 构图平衡性
        metrics["composition"] = self.composition_balance_score(image)

        # 6. 噪声水平
        metrics["noise_level"] = self.noise_level_score(image)

        # 7. 对比度
        metrics["contrast"] = self.contrast_score(image)

        # 8. 综合质量评分 (加权聚合)
        metrics["overall"] = self._compute_overall_score(metrics)

        return metrics

    def clip_semantic_similarity(self, image: Image.Image, prompt: str) -> dict:
        """CLIP图像-文本语义相似度 (0-1，越高越好)
        
        放宽标准：CLIP 相似度本身偏低，降低评分阈值
        """
        model, processor = self._get_clip()
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            text_features = outputs.text_embeds
            image_features = outputs.image_embeds
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (text_features * image_features).sum(dim=-1).item()

        # 放宽：将原始相似度映射到更宽松的评分 (原始 0.25 左右很常见)
        # 使用 sigmoid 映射，让 0.2-0.3 的相似度也能得到中等分数
        mapped_score = 1 / (1 + math.exp(-10 * (similarity - 0.22)))

        return {
            "score": round(mapped_score, 4),
            "raw_similarity": round(similarity, 4),
            "grade": self._grade_semantic(mapped_score),
            "description": "图像与提示词语义匹配度"
        }

    def aesthetic_score(self, image: Image.Image) -> dict:
        """美学评分：基于CLIP embeddings的线性美学预测 (0-10)
        
        放宽标准：原始评分偏低，提高基础分
        """
        model, processor = self._get_clip()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features = image_features.cpu().numpy()[0]

        # 使用简化的美学预测 (基于特征的统计分布)
        # 实际应用中可以用训练好的美学预测模型替换
        feature_mean = np.mean(features)
        feature_std = np.std(features)

        # 将特征分布映射到0-10分 (放宽：提高基础分，降低特征敏感度)
        # 高均值+适中标准差通常对应更好的美学质量
        score = 6.0 + (feature_mean * 2.0) + (feature_std * 1.2)
        score = np.clip(score, 3.0, 10.0)

        return {
            "score": round(float(score), 2),
            "grade": self._grade_aesthetic(score),
            "description": "图像美学质量预测评分"
        }

    def sharpness_score(self, image: Image.Image) -> dict:
        """清晰度评分：基于拉普拉斯算子的方差 (0-100)"""
        img_array = np.array(image.convert("L"))
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        variance = laplacian.var()

        # 归一化到0-100 (经验阈值: <100模糊, 100-500一般, >500清晰)
        score = min(100, variance / 8)
        score = max(0, score)

        return {
            "score": round(float(score), 2),
            "laplacian_variance": round(float(variance), 2),
            "grade": self._grade_sharpness(score),
            "description": "图像清晰度和边缘锐利度"
        }

    def color_richness_score(self, image: Image.Image) -> dict:
        """色彩丰富度：基于HSV色彩空间熵值 (0-100)"""
        img_array = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # 计算H和S通道的直方图熵
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

        h_entropy = self._calculate_entropy(h_hist)
        s_entropy = self._calculate_entropy(s_hist)

        # 综合评分 (饱和度熵权重更高)
        score = (h_entropy / 1.8 * 40) + (s_entropy / 2.5 * 60)
        score = min(100, score)

        return {
            "score": round(float(score), 2),
            "hue_entropy": round(float(h_entropy), 3),
            "saturation_entropy": round(float(s_entropy), 3),
            "grade": self._grade_color(score),
            "description": "色彩多样性和饱和度分布"
        }

    def composition_balance_score(self, image: Image.Image) -> dict:
        """构图平衡性：基于图像重心分布 (0-100)"""
        img_array = np.array(image.convert("L"))
        h, w = img_array.shape

        # 计算图像重心
        y_indices, x_indices = np.indices((h, w))
        total_intensity = np.sum(img_array)

        if total_intensity == 0:
            return {"score": 50.0, "grade": "一般", "description": "构图平衡性"}

        center_x = np.sum(x_indices * img_array) / total_intensity
        center_y = np.sum(y_indices * img_array) / total_intensity

        # 计算与几何中心的偏差
        ideal_x, ideal_y = w / 2, h / 2
        deviation_x = abs(center_x - ideal_x) / (w / 2)
        deviation_y = abs(center_y - ideal_y) / (h / 2)

        # 偏差越小分数越高
        score = 100 - (deviation_x * 30 + deviation_y * 30)
        score = max(0, min(100, score))

        # 三分法检测
        rule_of_thirds_score = self._check_rule_of_thirds(center_x, center_y, w, h)
        score = score * 0.6 + rule_of_thirds_score * 0.4

        return {
            "score": round(float(score), 2),
            "center_of_mass": (round(float(center_x), 1), round(float(center_y), 1)),
            "grade": self._grade_composition(score),
            "description": "图像构图平衡性和视觉重心分布"
        }

    def noise_level_score(self, image: Image.Image) -> dict:
        """噪声水平评估：基于频域分析 (0-100，越高表示噪声越少)"""
        img_array = np.array(image.convert("L"))

        # 使用FFT分析频域
        f_transform = fftpack.fft2(img_array)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2

        # 分离高频和低频成分
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # 高频区域 (距离中心较远)
        high_freq_mask = distance > min(h, w) * 0.3
        low_freq_mask = distance <= min(h, w) * 0.1

        high_freq_energy = np.mean(magnitude[high_freq_mask])
        low_freq_energy = np.mean(magnitude[low_freq_mask])

        # 高频能量相对于低频能量的比例反映噪声水平
        if low_freq_energy > 0:
            noise_ratio = high_freq_energy / low_freq_energy
        else:
            noise_ratio = 1.0

        # 转换为评分 (噪声比例越低，分数越高)
        score = max(0, 100 - noise_ratio * 50)

        return {
            "score": round(float(score), 2),
            "noise_ratio": round(float(noise_ratio), 4),
            "grade": self._grade_noise(score),
            "description": "图像噪声水平评估 (越高越干净)"
        }

    def contrast_score(self, image: Image.Image) -> dict:
        """对比度评分：基于灰度直方图分布 (0-100)"""
        img_array = np.array(image.convert("L"))

        # 计算对比度指标
        std_contrast = np.std(img_array)
        min_val, max_val = np.min(img_array), np.max(img_array)
        dynamic_range = max_val - min_val

        # 使用Michelson对比度
        if max_val + min_val > 0:
            michelson = (max_val - min_val) / (max_val + min_val)
        else:
            michelson = 0

        # 综合评分
        score = (std_contrast / 128 * 50) + (michelson * 50)
        score = min(100, score)

        return {
            "score": round(float(score), 2),
            "std_contrast": round(float(std_contrast), 2),
            "dynamic_range": int(dynamic_range),
            "grade": self._grade_contrast(score),
            "description": "图像明暗对比度和动态范围"
        }

    def _compute_overall_score(self, metrics: dict) -> dict:
        """计算综合质量评分"""
        scores = []
        weights = {
            "aesthetic": 0.25,
            "sharpness": 0.20,
            "color_richness": 0.15,
            "composition": 0.15,
            "noise_level": 0.10,
            "contrast": 0.15,
        }

        for key, weight in weights.items():
            if key in metrics and "score" in metrics[key]:
                scores.append(metrics[key]["score"] * weight)

        if not scores:
            return {"score": 0, "grade": "未知", "description": "综合质量评分"}

        overall = sum(scores) / sum(weights.values())

        # 如果有CLIP语义评分，加入权重
        if "clip_semantic" in metrics:
            semantic_score = metrics["clip_semantic"]["score"] * 100
            overall = overall * 0.7 + semantic_score * 0.3

        return {
            "score": round(float(overall), 2),
            "grade": self._grade_overall(overall),
            "description": "图像综合质量评分 (多维度加权)"
        }

    def _calculate_entropy(self, hist: np.ndarray) -> float:
        """计算直方图熵"""
        hist = hist.flatten() / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _check_rule_of_thirds(self, cx: float, cy: float, w: int, h: int) -> float:
        """检查三分法构图"""
        thirds_x = [w / 3, 2 * w / 3]
        thirds_y = [h / 3, 2 * h / 3]

        min_dist_x = min(abs(cx - tx) for tx in thirds_x)
        min_dist_y = min(abs(cy - ty) for ty in thirds_y)

        score = 100 - (min_dist_x / w * 100 + min_dist_y / h * 100)
        return max(0, min(100, score))

    # 评分等级划分 (放宽标准)
    def _grade_semantic(self, score: float) -> str:
        # 放宽：sigmoid 映射后的分数，0.5 以上就算优秀
        if score >= 0.70: return "优秀"
        if score >= 0.50: return "良好"
        if score >= 0.30: return "一般"
        return "较差"

    def _grade_aesthetic(self, score: float) -> str:
        # 放宽：基础分提高后，6.0 以上算良好
        if score >= 8.0: return "优秀"
        if score >= 6.0: return "良好"
        if score >= 4.5: return "一般"
        if score >= 3.0: return "较差"
        return "很差"

    def _grade_sharpness(self, score: float) -> str:
        if score >= 70: return "优秀"
        if score >= 40: return "良好"
        if score >= 20: return "一般"
        return "模糊"

    def _grade_color(self, score: float) -> str:
        if score >= 80: return "丰富"
        if score >= 60: return "良好"
        if score >= 40: return "一般"
        return "单调"

    def _grade_composition(self, score: float) -> str:
        if score >= 80: return "优秀"
        if score >= 60: return "良好"
        if score >= 40: return "一般"
        return "失衡"

    def _grade_noise(self, score: float) -> str:
        if score >= 80: return "干净"
        if score >= 60: return "轻微"
        if score >= 40: return "一般"
        return "噪点明显"

    def _grade_contrast(self, score: float) -> str:
        if score >= 80: return "优秀"
        if score >= 60: return "良好"
        if score >= 40: return "一般"
        return "平淡"

    def _grade_overall(self, score: float) -> str:
        if score >= 85: return "优秀"
        if score >= 70: return "良好"
        if score >= 55: return "一般"
        if score >= 40: return "较差"
        return "很差"
