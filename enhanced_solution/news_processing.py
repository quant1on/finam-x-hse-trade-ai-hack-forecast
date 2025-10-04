import pandas as pd
import numpy as np
from typing import Dict, List
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from config import Config
import warnings
import os

warnings.filterwarnings("ignore")

# Проверка доступности transformers для FinBERT
pipeline = None
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not installed. Enhanced mock version in use")
    pipeline = None


@dataclass
class CorporateEvent:
    """
    Структура данных для хранения информации о корпоративном событии компании.
    Используется для детекции и анализа влияния корпоративных действий на sentiment.
    """
    event_type: str
    ticker: str
    date: datetime
    ratio: float = None
    amount: float = None
    confidence: float = 0.0
    sentiment_impact: str = 'neutral'
    price_impact: str = 'none'
    description: str = ""


class TickerExtractor:
    """
    Класс для извлечения тикеров российских компаний из текстов финансовых новостей.
    Использует словари известных тикеров и соответствий названий компаний.
    """

    def __init__(self):
        """
                Инициализация экстрактора с базой данных российских тикеров и названий компаний.

                :return: None
                """
        self.known_tickers = Config.RUSSIAN_TICKERS['known_tickers']

        self.company_to_ticker = Config.RUSSIAN_TICKERS['company_to_ticker']

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """
        Извлечение всех упоминаний тикеров компаний из текста новости.

        :param text: Текст новости для анализа
        :return: Список найденных тикеров компаний
        """
        if pd.isna(text) or not text:
            return []

        found_tickers = set()
        text_lower = text.lower()

        for ticker in self.known_tickers:
            if ticker.lower() in text_lower:
                found_tickers.add(ticker)

        for company_name, ticker in self.company_to_ticker.items():
            if company_name in text_lower:
                found_tickers.add(ticker)

        return list(found_tickers)

    def extract_primary_ticker(self, text: str) -> str:
        """
        Извлечение основного (первого найденного) тикера из текста.
        Используется когда нужен один главный тикер для новости.

        :param text: Текст новости для анализа
        :return: Основной тикер компании или пустая строка
        """
        tickers = self.extract_tickers_from_text(text)
        return tickers[0] if tickers else ""


class CorporateEventsDetector:
    """
    Детектор корпоративных событий из финансовых новостей.
    Использует регулярные выражения для поиска дроблений, дивидендов, спин-оффов.
    """

    def __init__(self):
        """
                Инициализация детектора с паттернами для различных типов корпоративных событий.

                :return: None
                """
        self.split_patterns = {
            'ru': [
                r'дробление.*акций.*(\d+):(\d+)',
                r'сплит.*(\d+)\s*к\s*(\d+)',
                r'дробление.*(\d+)\s*на\s*(\d+)',
                r'разделение.*акций.*(\d+):(\d+)',
                r'stock.*split.*(\d+):(\d+)'
            ],
            'en': [
                r'stock.*split.*(\d+)[:\-](\d+)',
                r'share.*split.*(\d+)[:\-](\d+)',
                r'splitting.*(\d+)\s*for\s*(\d+)',
                r'(\d+)[:\-](\d+).*split'
            ]
        }

        self.dividend_patterns = {
            'ru': [
                r'дивиденд.*(\d+(?:\.\d+)?)\s*рубл',
                r'выплата.*дивидендов.*(\d+(?:\.\d+)?)',
                r'дивиденд.*размер.*(\d+(?:\.\d+)?)',
                r'дивидендные.*выплаты.*(\d+(?:\.\d+)?)',
                r'объявлен.*дивиденд.*(\d+(?:\.\d+)?)',
                r'дивиденд.*(\d+(?:\.\d+)?)\s*руб'
            ],
            'en': [
                r'dividend.*(\d+(?:\.\d+)?)\s*(?:rub|руб|ruble)',
                r'declared.*dividend.*(\d+(?:\.\d+)?)',
                r'quarterly.*dividend.*(\d+(?:\.\d+)?)',
                r'dividend.*payment.*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*dividend'
            ]
        }

        self.reverse_split_patterns = {
            'ru': [
                r'обратное.*дробление.*(\d+):(\d+)',
                r'консолидация.*акций.*(\d+):(\d+)',
                r'reverse.*split.*(\d+):(\d+)'
            ],
            'en': [
                r'reverse.*split.*(\d+)[:\-](\d+)',
                r'consolidation.*(\d+)[:\-](\d+)',
                r'(\d+)[:\-](\d+).*reverse.*split'
            ]
        }

        self.spinoff_patterns = {
            'ru': [
                r'спин[-\s]офф',
                r'выделение.*компании',
                r'отделение.*бизнеса',
                r'spin[-\s]off'
            ],
            'en': [
                r'spin[-\s]off',
                r'spin[-\s]out',
                r'corporate.*separation',
                r'business.*separation'
            ]
        }

    def detect_corporate_events(self, text: str, ticker: str = "",
                                publish_date: datetime = None) -> List[CorporateEvent]:
        """
        Комплексное обнаружение всех типов корпоративных событий в тексте новости.

        :param text: Полный текст новости для анализа
        :param ticker: Тикер компании (если известен)
        :param publish_date: Дата публикации новости
        :return: Список обнаруженных корпоративных событий
        """
        events = []
        text_lower = text.lower()

        # Детекция дроблений акций
        splits = self._detect_splits(text, text_lower, ticker, publish_date)
        events.extend(splits)

        # Детекция дивидендов
        dividends = self._detect_dividends(text, text_lower, ticker, publish_date)
        events.extend(dividends)

        # Детекция обратных дроблений
        reverse_splits = self._detect_reverse_splits(text, text_lower, ticker, publish_date)
        events.extend(reverse_splits)

        # Детекция спин-оффов
        spinoffs = self._detect_spinoffs(text, text_lower, ticker, publish_date)
        events.extend(spinoffs)

        return events

    def _detect_splits(self, text: str, text_lower: str, ticker: str,
                       publish_date: datetime) -> List[CorporateEvent]:
        """
        Детекция дроблений акций с извлечением коэффициентов.

        :param text: Исходный текст новости
        :param text_lower: Текст в нижнем регистре
        :param ticker: Тикер компании
        :param publish_date: Дата публикации
        :return: Список найденных дроблений акций
        """
        events = []

        for lang in ['ru', 'en']:
            for pattern in self.split_patterns[lang]:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    try:
                        if len(match.groups()) >= 2:
                            old_shares = int(match.group(1))
                            new_shares = int(match.group(2))

                            if new_shares > old_shares:
                                ratio = new_shares / old_shares
                                sentiment = 'positive'
                                price_impact = 'decrease'
                                description = f"Дробление акций {old_shares}:{new_shares}"
                            else:
                                ratio = old_shares / new_shares
                                sentiment = 'negative'
                                price_impact = 'increase'
                                description = f"Обратное дробление акций {old_shares}:{new_shares}"

                            event = CorporateEvent(
                                event_type='split',
                                ticker=ticker,
                                date=publish_date or datetime.now(),
                                ratio=ratio,
                                confidence=0.8,
                                sentiment_impact=sentiment,
                                price_impact=price_impact,
                                description=description
                            )
                            events.append(event)

                    except (ValueError, ZeroDivisionError):
                        continue

        return events

    def _detect_dividends(self, text: str, text_lower: str, ticker: str,
                          publish_date: datetime) -> List[CorporateEvent]:
        """
        Детекция выплат дивидендов с извлечением сумм.

        :param text: Исходный текст новости
        :param text_lower: Текст в нижнем регистре
        :param ticker: Тикер компании
        :param publish_date: Дата публикации
        :return: Список найденных дивидендных выплат
        """
        events = []

        for lang in ['ru', 'en']:
            for pattern in self.dividend_patterns[lang]:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    try:
                        amount = float(match.group(1))

                        sentiment = 'positive'
                        price_impact = 'decrease'

                        if amount > 50:
                            confidence = 0.9
                        elif amount > 10:
                            confidence = 0.8
                        else:
                            confidence = 0.7

                        event = CorporateEvent(
                            event_type='dividend',
                            ticker=ticker,
                            date=publish_date or datetime.now(),
                            amount=amount,
                            confidence=confidence,
                            sentiment_impact=sentiment,
                            price_impact=price_impact,
                            description=f"Выплата дивидендов {amount} руб."
                        )
                        events.append(event)

                    except ValueError:
                        continue

        return events

    def _detect_reverse_splits(self, text: str, text_lower: str, ticker: str,
                               publish_date: datetime) -> List[CorporateEvent]:
        """
        Детекция обратных дроблений акций (консолидации).

        :param text: Исходный текст новости
        :param text_lower: Текст в нижнем регистре
        :param ticker: Тикер компании
        :param publish_date: Дата публикации
        :return: Список найденных обратных дроблений
        """
        events = []

        for lang in ['ru', 'en']:
            for pattern in self.reverse_split_patterns[lang]:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    try:
                        old_shares = int(match.group(1))
                        new_shares = int(match.group(2))
                        ratio = old_shares / new_shares

                        event = CorporateEvent(
                            event_type='reverse_split',
                            ticker=ticker,
                            date=publish_date or datetime.now(),
                            ratio=ratio,
                            confidence=0.8,
                            sentiment_impact='negative',
                            price_impact='increase',
                            description=f"Обратное дробление акций {old_shares}:{new_shares}"
                        )
                        events.append(event)

                    except (ValueError, ZeroDivisionError):
                        continue

        return events

    def _detect_spinoffs(self, text: str, text_lower: str, ticker: str,
                         publish_date: datetime) -> List[CorporateEvent]:
        """
        Детекция спин-оффов (выделения дочерних компаний).

        :param text: Исходный текст новости
        :param text_lower: Текст в нижнем регистре
        :param ticker: Тикер компании
        :param publish_date: Дата публикации
        :return: Список найденных спин-оффов
        """
        events = []

        for lang in ['ru', 'en']:
            for pattern in self.spinoff_patterns[lang]:
                if re.search(pattern, text_lower):
                    sentiment = self._determine_context_sentiment(text_lower)

                    event = CorporateEvent(
                        event_type='spinoff',
                        ticker=ticker,
                        date=publish_date or datetime.now(),
                        confidence=0.7,
                        sentiment_impact=sentiment,
                        price_impact='none',
                        description="Выделение дочерней компании (спин-офф)"
                    )
                    events.append(event)
                    break

        return events

    def _determine_context_sentiment(self, text_lower: str) -> str:
        """
        Определение sentiment спин-оффа на основе контекстных слов.

        :param text_lower: Текст новости в нижнем регистре
        :return: Sentiment события ('positive', 'negative', 'neutral')
        """
        positive_keywords = ['рост', 'развитие', 'улучшение', 'фокус', 'эффективность', 'growth', 'improvement']
        negative_keywords = ['проблемы', 'трудности', 'кризис', 'снижение', 'problems', 'difficulties']

        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'


class FinBertSentimentAnalyzer:
    """
    FinBERT анализатор sentiment с интеграцией корпоративных событий.
    Использует предтренированную модель ProsusAI/finbert для анализа финансовых текстов.
    """

    def __init__(self):
        self.model_name = Config.FINBERT_CONFIG['model_name']
        self.max_length = Config.FINBERT_CONFIG['max_length']
        self.batch_size = Config.FINBERT_CONFIG['batch_size']
        self.device = Config.FINBERT_CONFIG['device']

        self.sentiment_pipeline = None
        self.corporate_detector = CorporateEventsDetector()
        self.ticker_extractor = TickerExtractor()
        self._initialize_finbert()

        # Коэффициенты корректировки sentiment для корпоративных событий
        self.corporate_sentiment_adjustments = {
            'split': 0.2,  # Дробления обычно позитивны
            'dividend': 0.3,  # Дивиденды очень позитивны
            'reverse_split': -0.4,  # Обратные дробления негативны
            'spinoff': 0.1  # Спин-оффы слабо позитивны
        }

    def _initialize_finbert(self):
        """
        Инициализация и загрузка FinBERT модели для sentiment анализа.

        :return: None
        """

        if not TRANSFORMERS_AVAILABLE:
            print("FinBERT unavailable, enhanced mock analysis in use")
            return

        try:
            print(f"Загружаем FinBERT: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,  # CPU
                return_all_scores=True
            )
            print("FinBERT успешно загружен")
        except Exception as e:
            print(f"Ошибка загрузки FinBERT: {e}")
            self.sentiment_pipeline = None

    def clean_financial_text(self, text: str) -> str:
        """
        Очистка и предобработка финансовых текстов для анализа.

        :param text: Исходный текст для очистки
        :return: Очищенный и нормализованный текст
        """

        if pd.isna(text) or text is None:
            return ""

        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b([A-Z]{2,5})\b', r' \1 ', text)

        # Ограничение длины для FinBERT
        if len(text) > self.max_length:
            text = text[:200] + " ... " + text[-200:]

        return text.strip()

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Базовый анализ sentiment текста через FinBERT или mock анализ.

        :param text: Текст для анализа sentiment
        :return: Словарь с вероятностями и метриками sentiment
        """
        cleaned_text = self.clean_financial_text(text)

        if not cleaned_text:
            return self._neutral_sentiment()

        if self.sentiment_pipeline is None:
            return self._enhanced_mock_sentiment_analysis(cleaned_text)

        try:
            results = self.sentiment_pipeline(cleaned_text)
            return self._process_finbert_results(results)
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return self._enhanced_mock_sentiment_analysis(cleaned_text)

    def analyze_sentiment_with_corporate_events(self, title: str, publication: str,
                                                tickers: List[str] = None,
                                                publish_date: datetime = None) -> Dict[str, float]:
        """
        Полный анализ sentiment с учетом корпоративных событий и корректировкой результата.

        :param title: Заголовок новости
        :param publication: Текст публикации
        :param tickers: Список тикеров компаний (извлекается автоматически если не указан)
        :param publish_date: Дата публикации новости

        :return: Расширенный словарь с sentiment и метриками корпоративных событий
        """
        full_text = f"{title} {publication}"

        if not tickers:
            primary_ticker = self.ticker_extractor.extract_primary_ticker(full_text)
            tickers = [primary_ticker] if primary_ticker else []

        base_sentiment = self.analyze_sentiment(full_text)

        all_corporate_events = []
        for ticker in tickers:
            events = self.corporate_detector.detect_corporate_events(
                full_text, ticker, publish_date
            )
            all_corporate_events.extend(events)

        if not all_corporate_events:
            base_sentiment.update({
                'has_corporate_events': False,
                'corporate_events_count': 0,
                'corporate_adjustment': 0.0,
                'tickers': tickers
            })
            return base_sentiment

        adjusted_sentiment = self._adjust_sentiment_for_corporate_events(
            base_sentiment, all_corporate_events
        )

        adjusted_sentiment.update({
            'has_corporate_events': True,
            'corporate_events_count': len(all_corporate_events),
            'corporate_event_types': [e.event_type for e in all_corporate_events],
            'max_corporate_confidence': max([e.confidence for e in all_corporate_events]),
            'tickers': tickers
        })

        return adjusted_sentiment

    def _adjust_sentiment_for_corporate_events(self, base_sentiment: Dict[str, float],
                                               corporate_events: List[CorporateEvent]) -> Dict[str, float]:
        """
        Корректировка базового sentiment на основе обнаруженных корпоративных событий.

        :param base_sentiment: Базовый sentiment от FinBERT
        :param corporate_events: Список корпоративных событий
        :return: Скорректированный sentiment с учетом корпоративных событий
        """

        adjusted = base_sentiment.copy()

        total_adjustment = 0.0
        max_confidence = 0.0

        for event in corporate_events:
            base_adjustment = self.corporate_sentiment_adjustments.get(event.event_type, 0.0)

            weighted_adjustment = base_adjustment * event.confidence

            if event.event_type == 'split' and event.ratio:
                if event.ratio > 2:
                    weighted_adjustment *= 1.5

            elif event.event_type == 'dividend' and event.amount:
                if event.amount > 20:
                    weighted_adjustment *= 1.3

            total_adjustment += weighted_adjustment
            max_confidence = max(max_confidence, event.confidence)

        original_score = adjusted['sentiment_score']
        adjusted_score = np.clip(original_score + total_adjustment, -1.0, 1.0)

        adjusted.update({
            'sentiment_score': adjusted_score,
            'original_sentiment_score': original_score,
            'corporate_adjustment': total_adjustment,
            'corporate_confidence': max_confidence
        })

        return adjusted

    def _enhanced_mock_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Улучшенный mock анализ sentiment с учетом финансовых терминов и корпоративных событий.

        :param text: Очищенный текст для анализа
        :return: Словарь с результатами mock анализа
        """

        positive_words = [
            'рост', 'прибыль', 'доход', 'успех', 'развитие', 'инвестиции',
            'дивиденд', 'дробление', 'расширение', 'улучшение', 'повышение',
            'укрепление', 'процветание', 'достижение', 'эффективность',
            'growth', 'profit', 'revenue', 'success', 'development', 'investment',
            'dividend', 'split', 'expansion', 'improvement', 'increase',
            'strengthen', 'prosperity', 'achievement', 'efficiency'
        ]

        negative_words = [
            'падение', 'убыток', 'кризис', 'проблемы', 'снижение', 'риск',
            'обратное дробление', 'консолидация', 'сокращение', 'спад',
            'ухудшение', 'потери', 'дефицит', 'банкротство', 'штрафы',
            'decline', 'loss', 'crisis', 'problems', 'decrease', 'risk',
            'reverse split', 'consolidation', 'reduction', 'downturn',
            'deterioration', 'losses', 'deficit', 'bankruptcy', 'penalties'
        ]

        text_lower = text.lower()

        # Специальная обработка корпоративных событий в mock режиме
        corporate_bonus = 0.0

        if any(word in text_lower for word in ['дивиденд', 'dividend']):
            corporate_bonus += 0.3
        if any(word in text_lower for word in
               ['дробление', 'split']) and 'обратное' not in text_lower and 'reverse' not in text_lower:
            corporate_bonus += 0.2
        if any(word in text_lower for word in ['обратное дробление', 'reverse split']):
            corporate_bonus -= 0.4

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return self._neutral_sentiment()

        base_score = (pos_count - neg_count) / total
        adjusted_score = np.clip(base_score + corporate_bonus, -1.0, 1.0)

        if adjusted_score > 0.1:
            return {
                'positive': 0.7 + abs(adjusted_score) * 0.2,
                'negative': 0.15, 'neutral': 0.15,
                'sentiment_score': adjusted_score,
                'confidence': 0.8
            }
        elif adjusted_score < -0.1:
            return {
                'positive': 0.15,
                'negative': 0.7 + abs(adjusted_score) * 0.2,
                'neutral': 0.15,
                'sentiment_score': adjusted_score,
                'confidence': 0.8
            }
        else:
            return {
                'positive': 0.35, 'negative': 0.35, 'neutral': 0.30,
                'sentiment_score': adjusted_score,
                'confidence': 0.6
            }

    def _neutral_sentiment(self) -> Dict[str, float]:
        """
        Возврат нейтрального sentiment по умолчанию.

        :return: Словарь с нейтральными значениями sentiment
        """

        return {
            'positive': 0.33, 'negative': 0.33, 'neutral': 0.34,
            'sentiment_score': 0.0, 'confidence': 0.5
        }

    def _process_finbert_results(self, results) -> Dict[str, float]:
        """
        обработка результатов FinBERT с правильной распаковкой вложенных списков.

        :param results: Результаты от FinBERT pipeline
        :return: Нормализованный словарь с sentiment метриками
        """

        sentiment_dict = {}

        # Проверка формата результатов FinBERT
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]

            for result in results:
                if isinstance(result, dict) and 'label' in result and 'score' in result:
                    label = result['label'].lower()
                    score = result['score']
                    sentiment_dict[label] = score
                else:
                    print(f"Неожиданный формат результата FinBERT: {result}")

        else:
            print(f"Неожиданный формат результатов FinBERT: {type(results)} - {results}")
            return self._neutral_sentiment()

        if not sentiment_dict:
            return self._neutral_sentiment()

        sentiment_score = (
                sentiment_dict.get('positive', 0) -
                sentiment_dict.get('negative', 0)
        )

        return {
            'positive': sentiment_dict.get('positive', 0),
            'negative': sentiment_dict.get('negative', 0),
            'neutral': sentiment_dict.get('neutral', 0),
            'sentiment_score': sentiment_score,
            'confidence': max(sentiment_dict.values()) if sentiment_dict else 0.5
        }


class NewsDataProcessor:
    """
    Главный процессор новостных данных для обработки и анализа финансовых новостей.
    Координирует загрузку, sentiment анализ и создание признаков.
    """

    def __init__(self, data_path: str = "data/raw/participants/"):
        """
        Инициализация процессора с путем к данным и компонентами анализа.

        :param data_path: Путь к папке с исходными новостными данными
        :return: None
        """
        self.data_path = data_path
        self.sentiment_analyzer = FinBertSentimentAnalyzer()
        self.ticker_extractor = TickerExtractor()

        print("News Data Processor инициализирован")
        print("FinBERT + корпоративные события")
        print("Автоматическое извлечение тикеров")

    def load_news_data(self, filename: str = "train_news.csv") -> pd.DataFrame:
        """
        Загрузка новостных данных из CSV файла с автоматической обработкой различных форматов.

        :param filename: Имя файла с новостными данными
        :return: DataFrame с колонками [publish_date, title, publication]
        """

        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            print(f"Файл не найден: {filepath}")
            return pd.DataFrame()

        try:
            news_df = pd.read_csv(filepath)

            if 'C1' in news_df.columns:
                news_df.rename(columns={
                    'C1': 'index',
                    'C2': 'publish_date',
                    'C3': 'title',
                    'C4': 'publication'
                }, inplace=True)

            print(f"Загружено {len(news_df)} новостей")

            required_cols = ['publish_date', 'title', 'publication']
            missing_cols = [col for col in required_cols if col not in news_df.columns]

            if missing_cols:
                print(f"Отсутствуют колонки: {missing_cols}")
                return pd.DataFrame()

            return news_df

        except Exception as e:
            print(f"Ошибка загрузки новостей: {e}")
            return pd.DataFrame()

    def process_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Полная обработка sentiment всех новостей с прогресс-индикатором и статистикой.

        :param news_df: DataFrame с исходными новостями
        :return: DataFrame с добавленными sentiment признаками и корпоративными событиями
        """

        if news_df.empty:
            print("Пустой DataFrame новостей")
            return pd.DataFrame()

        print(f"ОБРАБОТКА НОВОСТЕЙ + КОРПОРАТИВНЫЕ СОБЫТИЯ")
        print(f"Анализируем {len(news_df)} новостей...")

        processed_news = news_df.copy()

        sentiment_results = []

        for idx, row in processed_news.iterrows():
            title = str(row.get('title', ''))
            publication = str(row.get('publication', ''))

            publish_date = None
            if 'publish_date' in row:
                try:
                    publish_date = pd.to_datetime(row['publish_date'])
                except:
                    pass

            sentiment_result = self.sentiment_analyzer.analyze_sentiment_with_corporate_events(
                title, publication, publish_date=publish_date
            )

            sentiment_score = sentiment_result['sentiment_score']
            if sentiment_score > 0.15:
                final_label = 'positive'
            elif sentiment_score < -0.15:
                final_label = 'negative'
            else:
                final_label = 'neutral'

            sentiment_results.append({
                'sentiment_score': sentiment_score,
                'confidence': sentiment_result['confidence'],
                'sentiment_label': final_label,
                'positive_prob': sentiment_result['positive'],
                'negative_prob': sentiment_result['negative'],
                'neutral_prob': sentiment_result['neutral'],

                # Корпоративные события
                'corporate_adjustment': sentiment_result.get('corporate_adjustment', 0.0),
                'has_corporate_events': sentiment_result.get('has_corporate_events', False),
                'corporate_events_count': sentiment_result.get('corporate_events_count', 0),
                'max_corporate_confidence': sentiment_result.get('max_corporate_confidence', 0.0),
                'tickers': ','.join(sentiment_result.get('tickers', []))
            })

            # Прогресс
            if (idx + 1) % 100 == 0:
                print(f"   Обработано: {idx + 1}/{len(processed_news)}")

        sentiment_df = pd.DataFrame(sentiment_results)
        final_result = pd.concat([processed_news, sentiment_df], axis=1)

        print(f"Обработка новостей завершена!")
        self._print_sentiment_stats(sentiment_df)

        return final_result

    def _print_sentiment_stats(self, sentiment_df: pd.DataFrame):
        """
        Вывод подробной статистики по результатам sentiment анализа.

        :param sentiment_df: DataFrame с результатами sentiment анализа
        :return: None
        """
        stats = sentiment_df['sentiment_label'].value_counts()
        total = len(sentiment_df)

        print("Статистика sentiment:")
        for label, count in stats.items():
            print(f"   {label}: {count} ({count / total * 100:.1f}%)")

        avg_confidence = sentiment_df['confidence'].mean()
        avg_corporate_adj = sentiment_df['corporate_adjustment'].mean()
        corporate_events_total = sentiment_df['corporate_events_count'].sum()

        print(f"   Средняя уверенность: {avg_confidence:.3f}")
        print(f"   Средняя корректировка: {avg_corporate_adj:.3f}")
        print(f"   Всего корпоративных событий: {int(corporate_events_total)}")


class SentimentAggregator:
    """
    Агрегатор sentiment данных для создания дневных признаков по тикерам.
    Обрабатывает временные ряды и создает скользящие признаки.
    """

    def __init__(self):
        """
        Инициализация агрегатора sentiment данных.
        """
        self.rolling_windows = Config.SENTIMENT_CONFIG['rolling_windows']
        self.confidence_threshold = Config.SENTIMENT_CONFIG['confidence_threshold']
        self.corporate_events_weight = Config.SENTIMENT_CONFIG['corporate_events_weight']

    def parse_tickers_string(self, tickers_str: str) -> List[str]:
        """
        Парсинг строки тикеров с различными разделителями.

        :param tickers_str: Строка с тикерами, разделенными запятыми, точками с запятой и др.
        :return: Список отдельных тикеров
        """
        if pd.isna(tickers_str) or not tickers_str:
            return []

        separators = [',', ';', '|', ' ']
        tickers = [tickers_str]

        for sep in separators:
            new_tickers = []
            for ticker in tickers:
                new_tickers.extend([t.strip() for t in str(ticker).split(sep) if t.strip()])
            tickers = new_tickers

        return [t for t in tickers if t and len(t) > 1]

    def aggregate_daily_sentiment(self, processed_news: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегация sentiment по дням и тикерам с расчетом статистик.

        :param processed_news: DataFrame с обработанными новостями
        :return: DataFrame с дневными агрегациями по тикерам
        """

        if processed_news.empty:
            return pd.DataFrame()

        print("Агрегация sentiment по дням и тикерам...")

        df = processed_news.copy()
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['date'] = df['publish_date'].dt.date

        expanded_rows = []
        for idx, row in df.iterrows():
            tickers = self.parse_tickers_string(row.get('tickers', ''))
            if not tickers:
                continue

            for ticker in tickers:
                new_row = row.copy()
                new_row['ticker'] = ticker
                expanded_rows.append(new_row)

        if not expanded_rows:
            print("Нет новостей с тикерами для агрегации")
            return pd.DataFrame()

        expanded_df = pd.DataFrame(expanded_rows)

        print(f"   Новостей с тикерами: {len(expanded_df)}")
        print(f"   Уникальных тикеров: {expanded_df['ticker'].nunique()}")

        agg_dict = {
            'sentiment_score': ['mean', 'std', 'count'],
            'confidence': 'mean',
            'sentiment_label': [
                lambda x: (x == 'positive').sum(),
                lambda x: (x == 'negative').sum(),
                lambda x: (x == 'neutral').sum()
            ],
            'corporate_adjustment': 'mean',
            'corporate_events_count': 'sum',
            'has_corporate_events': lambda x: (x == True).sum()
        }

        daily_agg = expanded_df.groupby(['date', 'ticker']).agg(agg_dict).reset_index()

        if not daily_agg.empty:
            daily_agg.columns = [
                'date', 'ticker', 'sentiment_mean', 'sentiment_std', 'news_count',
                'confidence_mean', 'positive_count', 'negative_count', 'neutral_count',
                'corporate_adjustment_mean', 'total_corporate_events', 'news_with_corporate_events'
            ]

            daily_agg['sentiment_std'] = daily_agg['sentiment_std'].fillna(0)
            daily_agg['positive_ratio'] = daily_agg['positive_count'] / daily_agg['news_count']
            daily_agg['negative_ratio'] = daily_agg['negative_count'] / daily_agg['news_count']
            daily_agg['corporate_events_ratio'] = daily_agg['news_with_corporate_events'] / daily_agg['news_count']

        print(f"Создано {len(daily_agg)} дневных агрегаций")
        return daily_agg

    def create_sentiment_features(self, daily_sentiment: pd.DataFrame,
                                  lag_days: int = 1) -> pd.DataFrame:
        """
        Создание sentiment признаков с задержкой для учета влияния новостей на следующий день.

        :param daily_sentiment: DataFrame с дневными агрегациями
        :param lag_days: Количество дней задержки влияния новостей
        :return: DataFrame с temporal признаками
        """

        if daily_sentiment.empty:
            return pd.DataFrame()

        df = daily_sentiment.copy()
        df['date'] = pd.to_datetime(df['date']) + timedelta(days=lag_days)

        df_enhanced = self._create_rolling_sentiment_features(df)

        return df_enhanced

    def _create_rolling_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление скользящих временных признаков для каждого тикера.

        :param df: DataFrame с дневными sentiment данными
        :return: DataFrame с добавленными скользящими признаками
        """

        df_sorted = df.sort_values(['ticker', 'date'])

        for ticker in df['ticker'].unique():
            mask = df_sorted['ticker'] == ticker
            ticker_data = df_sorted[mask]

            for window in self.rolling_windows:
                col_name = f'sentiment_ma_{window}'
                if col_name not in df_sorted.columns:
                    df_sorted[col_name] = 0.0

                rolling_mean = ticker_data['sentiment_mean'].rolling(
                    window=window, min_periods=1
                ).mean()

                df_sorted.loc[mask, col_name] = rolling_mean

            if 'sentiment_momentum' not in df_sorted.columns:
                df_sorted['sentiment_momentum'] = 0.0

            momentum = (
                    ticker_data['sentiment_mean'] -
                    ticker_data['sentiment_mean'].rolling(3, min_periods=1).mean()
            )
            df_sorted.loc[mask, 'sentiment_momentum'] = momentum

            if 'sentiment_volatility' not in df_sorted.columns:
                df_sorted['sentiment_volatility'] = 0.0

            volatility = ticker_data['sentiment_mean'].rolling(
                window=7, min_periods=1
            ).std()
            df_sorted.loc[mask, 'sentiment_volatility'] = volatility.fillna(0)

            if 'corporate_events_ma_7' not in df_sorted.columns:
                df_sorted['corporate_events_ma_7'] = 0.0

            corporate_ma = ticker_data['total_corporate_events'].rolling(
                window=7, min_periods=1
            ).sum()
            df_sorted.loc[mask, 'corporate_events_ma_7'] = corporate_ma

        return df_sorted.fillna(0)


def process_news_for_patchtst(data_path: str = Config.DATA_PATHS['raw_participants'],
                              news_filename: str = Config.DATA_FILES['news']) -> pd.DataFrame:
    """
    ГЛАВНАЯ ФУНКЦИЯ обработки новостей для PatchTST модели.
    Координирует весь пайплайн от загрузки до создания признаков.

    :param data_path: Путь к папке с исходными данными
    :param news_filename: Имя файла с новостными данными
    :return: DataFrame с готовыми sentiment признаками для PatchTST
    """
    print("ПАЙПЛАЙН ОБРАБОТКИ НОВОСТЕЙ - ЧАСТЬ B")
    print("=" * 60)
    print("FinBERT sentiment анализ")
    print("Детекция дроблений акций")
    print("Анализ дивидендов")
    print("Обратные дробления и спин-оффы")
    print("Корректировка sentiment для корпоративных событий")
    print("Интеграция с PatchTST данными")
    print("=" * 60)

    processor = NewsDataProcessor(data_path)

    news_df = processor.load_news_data(news_filename)

    if news_df.empty:
        print("Обработка новостей не удалась")
        return pd.DataFrame()

    processed_news = processor.process_news_sentiment(news_df)

    aggregator = SentimentAggregator()
    daily_sentiment = aggregator.aggregate_daily_sentiment(processed_news)

    if daily_sentiment.empty:
        print("Не удалось создать дневные агрегации, возвращаем обработанные новости")
        return processed_news

    sentiment_features = aggregator.create_sentiment_features(daily_sentiment)

    output_dir = data_path.replace('raw', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    sentiment_output_path = os.path.join(output_dir, Config.DATA_FILES['processed_sentiment'])
    sentiment_features.to_csv(sentiment_output_path, index=False)

    news_output_path = os.path.join(output_dir, Config.DATA_FILES['processed_news_with_corporate_events'])
    processed_news.to_csv(news_output_path, index=False)

    print(f"Результаты сохранены:")
    print(f"{sentiment_output_path}")
    print(f"{news_output_path}")
    print(f"Итого sentiment признаков: {len(sentiment_features.columns)}")

    if not sentiment_features.empty:
        corporate_events_total = sentiment_features['total_corporate_events'].sum()
        print(f"Корпоративных событий: {int(corporate_events_total)}")

    print("\nГОТОВО К ИНТЕГРАЦИИ С PATCHTST!")

    return sentiment_features


if __name__ == "__main__":
    result = process_news_for_patchtst()

    if not result.empty:
        print(f"Создано {len(result)} записей с sentiment признаками")
    else:
        print("\nОбработка новостей не удалась")