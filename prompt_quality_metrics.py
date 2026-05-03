import re
from typing import Dict, List, Tuple


class PromptQualityMetrics:
    """提示词质量量化评估体系 (适配 DeepSeek 扩充 + CLIP 长度限制)"""

    # 艺术风格关键词库
    STYLE_KEYWORDS = {
        "绘画风格": ["油画", "水彩", "素描", "速写", "国画", "工笔画", "写意", "水墨",
                   "oil painting", "watercolor", "sketch", "pencil", "ink", "pastel"],
        "艺术流派": ["印象派", "抽象", "写实", "超现实", "立体主义", "波普艺术",
                   "impressionism", "abstract", "realistic", "surrealism", "cubism", "pop art"],
        "渲染风格": ["3D", "CG", "像素", "低多边形", "卡通", "动漫", "二次元",
                   "3d render", "cg", "pixel art", "low poly", "cartoon", "anime"],
    }

    # 光线关键词
    LIGHTING_KEYWORDS = [
        "自然光", "柔和光", "硬光", "逆光", "侧光", "顶光", "底光",
        "golden hour", "soft light", "hard light", "backlight", "side light",
        "volumetric", "cinematic lighting", "dramatic", "neon", "ambient",
        "studio lighting", "rim light", "bloom", "ray tracing"
    ]

    # 构图关键词
    COMPOSITION_KEYWORDS = [
        "特写", "中景", "全景", "鸟瞰", "仰视", "俯视",
        "close-up", "medium shot", "wide shot", "aerial view", "low angle", "high angle",
        "portrait", "landscape", "rule of thirds", "symmetry", "leading lines",
        "depth of field", "bokeh", "foreground", "background"
    ]

    # 质量增强关键词
    QUALITY_KEYWORDS = [
        "高质量", "高分辨率", "精细", "细节丰富", "8k", "4k",
        "high quality", "high resolution", "detailed", "intricate", "masterpiece",
        "best quality", "ultra detailed", "sharp focus", "hdr", "8k resolution"
    ]

    # 负面/模糊词汇
    VAGUE_WORDS = [
        "东西", "物品", "画面", "图", "样子", "感觉", "氛围",
        "thing", "stuff", "something", "nice", "good", "beautiful", "cool"
    ]

    def evaluate_all(self, prompt: str, enhanced_prompt: str = "") -> dict:
        """运行所有提示词质量评估指标，返回完整的评估报告。
        
        改进点：
        1. 如果有 enhanced_prompt，优先对 enhanced_prompt 评分
        2. 放宽信息密度和长度合理性标准（考虑 CLIP 70 字限制）
        """
        # 优先使用扩充后的提示词进行评估
        target_prompt = enhanced_prompt.strip() if enhanced_prompt and enhanced_prompt.strip() else prompt
        
        metrics = {}

        # 1. 信息密度 (放宽标准)
        metrics["information_density"] = self.information_density_score(target_prompt)

        # 2. 结构化程度
        metrics["structure"] = self.structure_completeness_score(target_prompt)

        # 3. 语言纯净度
        metrics["language_purity"] = self.language_purity_score(target_prompt)

        # 4. 具体性评分
        metrics["specificity"] = self.specificity_score(target_prompt)

        # 5. 风格完整性
        metrics["style_completeness"] = self.style_completeness_score(target_prompt)

        # 6. 长度合理性 (放宽标准，适配 CLIP 限制)
        metrics["length_appropriateness"] = self.length_appropriateness_score(target_prompt)

        # 7. 质量关键词使用
        metrics["quality_boost"] = self.quality_boost_score(target_prompt)

        # 8. 增强效果评估 (如果有增强后的提示词)
        if enhanced_prompt and enhanced_prompt != prompt:
            metrics["enhancement_effect"] = self.enhancement_effect_score(prompt, enhanced_prompt)

        # 9. 综合质量评分
        metrics["overall"] = self._compute_overall_score(metrics)

        return metrics

    def information_density_score(self, prompt: str) -> dict:
        """信息密度：关键词数量、多样性和独特性 (0-100)
        
        放宽标准：DeepSeek 扩充后的提示词通常较长，降低密度要求
        """
        tokens = self._tokenize_prompt(prompt)

        if not tokens:
            return {"score": 0, "grade": "很差", "description": "提示词信息密度"}

        # 去重后的关键词数量
        unique_tokens = set(t.lower() for t in tokens)

        # 计算多样性 (不同词性/类别的覆盖)
        categories_covered = self._count_categories(tokens)

        # 密度评分 (放宽：降低 unique_tokens 权重，提高类别覆盖权重)
        density = len(unique_tokens) / max(len(tokens), 1)
        # 放宽：更多 tokens 也能获得较高分数
        score = (len(unique_tokens) * 3) + (categories_covered * 15) + (density * 15) + 30
        score = min(100, score)

        return {
            "score": round(float(score), 2),
            "total_keywords": len(tokens),
            "unique_keywords": len(unique_tokens),
            "categories_covered": categories_covered,
            "grade": self._grade_density(score),
            "description": "提示词关键词数量和多样性"
        }

    def structure_completeness_score(self, prompt: str) -> dict:
        """结构化程度：是否包含主体/风格/光线/构图等要素 (0-100)"""
        prompt_lower = prompt.lower()

        # 检查各个维度
        dimensions = {
            "主体描述": self._has_subject(prompt),
            "艺术风格": self._has_style(prompt),
            "光线氛围": self._has_lighting(prompt),
            "构图视角": self._has_composition(prompt),
            "色彩描述": self._has_color(prompt),
            "质感细节": self._has_texture(prompt),
        }

        covered = sum(1 for v in dimensions.values() if v)
        total = len(dimensions)

        # 基础分 + 完整度奖励 (放宽：3个维度即可得高分)
        base_score = (covered / total) * 60
        bonus = 40 if covered >= 4 else (25 if covered >= 3 else (10 if covered >= 2 else 0))
        score = base_score + bonus

        return {
            "score": round(float(score), 2),
            "dimensions": dimensions,
            "dimensions_covered": f"{covered}/{total}",
            "grade": self._grade_structure(score),
            "description": "提示词结构完整性 (主体/风格/光线/构图等)"
        }

    def language_purity_score(self, prompt: str) -> dict:
        """语言纯净度：检测中英文混杂程度 (0-100，越高越纯净)"""
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', prompt))
        has_english = bool(re.search(r'[a-zA-Z]', prompt))

        if not has_chinese and not has_english:
            return {"score": 0, "grade": "很差", "description": "语言纯净度"}

        # 检测混合模式
        mixed_patterns = len(re.findall(r'[\u4e00-\u9fff][a-zA-Z]|[a-zA-Z][\u4e00-\u9fff]', prompt))
        total_chars = len(prompt)

        if total_chars > 0:
            mix_ratio = mixed_patterns / total_chars
        else:
            mix_ratio = 0

        # 纯净度评分 (放宽：DeepSeek 输出通常是英文，允许纯英文)
        if has_chinese and has_english:
            # 中英混合，根据混合程度扣分 (放宽阈值)
            score = max(0, 100 - mix_ratio * 300)
        else:
            # 纯中文或纯英文 (DeepSeek 通常是纯英文，给满分)
            score = 100

        return {
            "score": round(float(score), 2),
            "has_chinese": has_chinese,
            "has_english": has_english,
            "mix_ratio": round(float(mix_ratio), 4),
            "grade": self._grade_purity(score),
            "description": "语言纯净度 (越低表示中英文混杂越严重)"
        }

    def specificity_score(self, prompt: str) -> dict:
        """具体性评分：抽象 vs 具体描述比例 (0-100)"""
        tokens = self._tokenize_prompt(prompt)

        if not tokens:
            return {"score": 0, "grade": "很差", "description": "具体性评分"}

        vague_count = sum(1 for t in tokens if t.lower() in self.VAGUE_WORDS)
        specific_indicators = [
            "detailed", "intricate", "specific", "precise", "定义的",
            "详细的", "精确的", "清晰的", "具体的"
        ]
        specific_count = sum(1 for t in tokens if any(si in t.lower() for si in specific_indicators))

        total = len(tokens)
        vague_ratio = vague_count / total if total > 0 else 0

        # 具体性评分 (放宽：降低模糊词惩罚)
        score = 100 - (vague_ratio * 50) + (specific_count * 3) + 20
        score = max(0, min(100, score))

        return {
            "score": round(float(score), 2),
            "vague_words": vague_count,
            "specific_indicators": specific_count,
            "grade": self._grade_specificity(score),
            "description": "描述具体程度 (越少模糊词汇越好)"
        }

    def style_completeness_score(self, prompt: str) -> dict:
        """风格完整性：是否包含完整的艺术风格描述 (0-100)"""
        prompt_lower = prompt.lower()

        style_coverage = {}
        total_keywords = 0
        matched_keywords = 0

        for category, keywords in self.STYLE_KEYWORDS.items():
            category_match = sum(1 for kw in keywords if kw.lower() in prompt_lower)
            style_coverage[category] = category_match > 0
            total_keywords += len(keywords)
            matched_keywords += min(category_match, 2)  # 每个类别最多计2分

        # 风格覆盖率 (放宽：有1个类别即可得基础分)
        categories_covered = sum(1 for v in style_coverage.values() if v)
        coverage_score = (categories_covered / len(self.STYLE_KEYWORDS)) * 50 + 30

        # 关键词丰富度
        richness_score = (matched_keywords / max(total_keywords * 0.1, 1)) * 30

        score = min(100, coverage_score + richness_score)

        return {
            "score": round(float(score), 2),
            "style_coverage": style_coverage,
            "categories_covered": categories_covered,
            "grade": self._grade_style(score),
            "description": "艺术风格描述完整性"
        }

    def length_appropriateness_score(self, prompt: str) -> dict:
        """长度合理性：字数是否在最佳区间 (0-100)
        
        放宽标准：适配 CLIP 70 字限制，允许较短提示词
        """
        # 去除空白后的长度
        clean_prompt = prompt.strip()
        char_count = len(clean_prompt)

        # 放宽最佳区间：20-80字 (适配 CLIP 限制)
        # DeepSeek 扩充后通常在 50-150 字，但 CLIP 只取前 70 字左右
        optimal_min, optimal_max = 20, 80

        if char_count < optimal_min:
            # 太短 (放宽：最低给 50 分)
            score = 50 + (char_count / optimal_min) * 30
        elif char_count <= optimal_max:
            # 最佳区间
            score = 100
        else:
            # 太长，适当扣分 (放宽：超过 80 字才开始扣分)
            excess = char_count - optimal_max
            score = max(50, 100 - excess * 0.5)

        return {
            "score": round(float(score), 2),
            "char_count": char_count,
            "optimal_range": f"{optimal_min}-{optimal_max}字",
            "grade": self._grade_length(score),
            "description": "提示词长度合理性 (适配 CLIP 限制)"
        }

    def quality_boost_score(self, prompt: str) -> dict:
        """质量增强关键词使用 (0-100)"""
        prompt_lower = prompt.lower()

        matched = []
        for kw in self.QUALITY_KEYWORDS:
            if kw.lower() in prompt_lower:
                matched.append(kw)

        # 去重计数
        unique_matched = list(set(matched))
        count = len(unique_matched)

        # 适量使用质量词有益，过多可能冗余 (放宽)
        if count == 0:
            score = 60  # 没有质量词，基础分提高
        elif count <= 3:
            score = 75 + count * 8  # 适量使用
        elif count <= 6:
            score = 100  # 最佳
        else:
            score = max(60, 100 - (count - 6) * 3)  # 过多扣分减少

        return {
            "score": round(float(score), 2),
            "quality_keywords_used": unique_matched,
            "count": count,
            "grade": self._grade_quality_boost(score),
            "description": "质量增强关键词使用合理性"
        }

    def enhancement_effect_score(self, original: str, enhanced: str) -> dict:
        """评估提示词增强效果 (0-100)"""
        orig_metrics = self.evaluate_all(original)
        enhanced_metrics = self.evaluate_all(enhanced)

        # 比较各维度提升
        improvements = {}
        total_improvement = 0
        count = 0

        compare_keys = ["information_density", "structure", "specificity", "style_completeness"]
        for key in compare_keys:
            if key in orig_metrics and key in enhanced_metrics:
                orig_score = orig_metrics[key]["score"]
                enh_score = enhanced_metrics[key]["score"]
                delta = enh_score - orig_score
                improvements[key] = round(float(delta), 2)
                total_improvement += delta
                count += 1

        avg_improvement = total_improvement / count if count > 0 else 0

        # 增强效果评分 (放宽)
        if avg_improvement > 15:
            score = 100
        elif avg_improvement > 8:
            score = 80 + (avg_improvement - 8) * 1.5
        elif avg_improvement > 0:
            score = 65 + avg_improvement * 2
        else:
            score = max(0, 65 + avg_improvement * 1.5)

        return {
            "score": round(float(score), 2),
            "improvements": improvements,
            "avg_improvement": round(float(avg_improvement), 2),
            "grade": self._grade_enhancement(score),
            "description": "提示词增强效果评估"
        }

    def _compute_overall_score(self, metrics: dict) -> dict:
        """计算提示词综合质量评分 (调整权重)"""
        # 降低信息密度和长度合理性权重，提高结构完整性权重
        weights = {
            "information_density": 0.15,      # 降低
            "structure": 0.30,                # 提高
            "language_purity": 0.10,
            "specificity": 0.15,
            "style_completeness": 0.15,
            "length_appropriateness": 0.05,   # 降低
            "quality_boost": 0.10,            # 提高
        }

        scores = []
        for key, weight in weights.items():
            if key in metrics and "score" in metrics[key]:
                scores.append(metrics[key]["score"] * weight)

        if not scores:
            return {"score": 0, "grade": "未知", "description": "提示词综合质量评分"}

        overall = sum(scores) / sum(weights.values())

        return {
            "score": round(float(overall), 2),
            "grade": self._grade_overall(overall),
            "description": "提示词综合质量评分 (多维度加权)"
        }

    def _tokenize_prompt(self, prompt: str) -> List[str]:
        """简单分词：按逗号、顿号分隔"""
        # 替换常见分隔符
        normalized = prompt.replace("，", ",").replace("、", ",")
        tokens = [t.strip() for t in normalized.split(",") if t.strip()]
        return tokens

    def _count_categories(self, tokens: List[str]) -> int:
        """统计覆盖的类别数"""
        categories = set()
        all_keywords = {}
        all_keywords.update({k: "风格" for v in self.STYLE_KEYWORDS.values() for k in v})
        all_keywords.update({k: "光线" for k in self.LIGHTING_KEYWORDS})
        all_keywords.update({k: "构图" for k in self.COMPOSITION_KEYWORDS})
        all_keywords.update({k: "质量" for k in self.QUALITY_KEYWORDS})

        for token in tokens:
            token_lower = token.lower()
            for kw, cat in all_keywords.items():
                if kw.lower() in token_lower:
                    categories.add(cat)

        return len(categories)

    def _has_subject(self, prompt: str) -> bool:
        """检查是否有主体描述"""
        # 简单启发式：如果包含名词性描述
        subject_indicators = [
            "一个", "一位", "一只", "场景", "人物", "女孩", "男孩", "女人", "男人",
            "a ", "an ", "the ", "portrait of", "image of", "scene of",
            "character", "person", "girl", "boy", "woman", "man"
        ]
        return any(si in prompt.lower() for si in subject_indicators)

    def _has_style(self, prompt: str) -> bool:
        """检查是否有风格描述"""
        all_styles = [k for v in self.STYLE_KEYWORDS.values() for k in v]
        return any(s.lower() in prompt.lower() for s in all_styles)

    def _has_lighting(self, prompt: str) -> bool:
        """检查是否有光线描述"""
        return any(l.lower() in prompt.lower() for l in self.LIGHTING_KEYWORDS)

    def _has_composition(self, prompt: str) -> bool:
        """检查是否有构图描述"""
        return any(c.lower() in prompt.lower() for c in self.COMPOSITION_KEYWORDS)

    def _has_color(self, prompt: str) -> bool:
        """检查是否有色彩描述"""
        color_words = [
            "红", "蓝", "绿", "黄", "紫", "橙", "粉", "白", "黑", "金", "银",
            "red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black",
            "colorful", "vibrant", "muted", "monochrome", "warm", "cool", "色调"
        ]
        return any(c.lower() in prompt.lower() for c in color_words)

    def _has_texture(self, prompt: str) -> bool:
        """检查是否有质感/细节描述"""
        texture_words = [
            "纹理", "质感", "细节", "光滑", "粗糙", "金属", "玻璃", "布料",
            "texture", "detailed", "smooth", "rough", "metallic", "glass", "fabric",
            "skin", "hair", "fur", "scale", "pattern"
        ]
        return any(t.lower() in prompt.lower() for t in texture_words)

    # 评分等级划分 (放宽)
    def _grade_density(self, score: float) -> str:
        if score >= 75: return "丰富"
        if score >= 55: return "良好"
        if score >= 35: return "一般"
        return "稀疏"

    def _grade_structure(self, score: float) -> str:
        if score >= 75: return "完整"
        if score >= 55: return "良好"
        if score >= 35: return "一般"
        return "缺失"

    def _grade_purity(self, score: float) -> str:
        if score >= 85: return "纯净"
        if score >= 60: return "轻微混杂"
        if score >= 40: return "混杂"
        return "严重混杂"

    def _grade_specificity(self, score: float) -> str:
        if score >= 75: return "具体"
        if score >= 55: return "较具体"
        if score >= 35: return "一般"
        return "模糊"

    def _grade_style(self, score: float) -> str:
        if score >= 75: return "完整"
        if score >= 55: return "良好"
        if score >= 35: return "一般"
        return "缺失"

    def _grade_length(self, score: float) -> str:
        if score >= 85: return "合适"
        if score >= 60: return "稍长/短"
        if score >= 40: return "偏长/短"
        return "不合适"

    def _grade_quality_boost(self, score: float) -> str:
        if score >= 85: return "优秀"
        if score >= 65: return "良好"
        if score >= 45: return "一般"
        return "不足"

    def _grade_enhancement(self, score: float) -> str:
        if score >= 85: return "显著提升"
        if score >= 65: return "有效提升"
        if score >= 45: return "轻微提升"
        return "效果有限"

    def _grade_overall(self, score: float) -> str:
        if score >= 80: return "优秀"
        if score >= 65: return "良好"
        if score >= 50: return "一般"
        if score >= 35: return "较差"
        return "很差"
