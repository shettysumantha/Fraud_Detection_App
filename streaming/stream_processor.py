import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ml_pipeline.predict import predict_transaction

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None
    KAFKA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class StreamProcessor:
    def __init__(
        self,
        model_path: Path = Path("models/fraud_pipeline.joblib"),
        history_path: Path = Path("data/processed_transactions.csv"),
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "fraud_transactions",
        output_topic: str = "fraud_alerts",
        alert_threshold: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.history_path = history_path
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.alert_threshold = alert_threshold
        self.producer = None

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        result = predict_transaction(record, model_path=self.model_path, history_path=self.history_path)
        self.publish_alert(result, record)
        return result

    def publish_alert(self, result: Dict[str, Any], record: Dict[str, Any]) -> None:
        alert_payload = {
            "transaction": record,
            "fraud_probability": result["fraud_probability"],
            "label": result["label"],
        }

        if result["fraud_probability"] >= self.alert_threshold or result["label"] == 1:
            logging.warning("Fraud alert generated: %s", json.dumps(alert_payload))
            if KAFKA_AVAILABLE and self.producer is not None:
                self.producer.send(self.output_topic, json.dumps(alert_payload).encode("utf-8"))
                self.producer.flush()

    def run_kafka(self) -> None:
        if not KAFKA_AVAILABLE:
            raise RuntimeError("Kafka client library is not installed. Install kafka-python to enable Kafka streaming.")

        consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)

        logging.info("Listening for stream events on topic %s", self.input_topic)
        for message in consumer:
            record = message.value
            self.process_record(record)

    def run_batch(self, input_path: Path) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"Stream batch file not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                self.process_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream processor skeleton for fraud detection.")
    parser.add_argument("--mode", choices=["kafka", "batch"], default="batch", help="Stream mode to run.")
    parser.add_argument("--input-file", default="data/stream_events.jsonl", help="Batch input file path for JSON lines.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers.")
    parser.add_argument("--input-topic", default="fraud_transactions", help="Kafka input topic.")
    parser.add_argument("--output-topic", default="fraud_alerts", help="Kafka output alert topic.")
    parser.add_argument("--alert-threshold", type=float, default=0.7, help="Fraud probability threshold for alerts.")
    args = parser.parse_args()

    processor = StreamProcessor(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        alert_threshold=args.alert_threshold,
    )

    if args.mode == "kafka":
        processor.run_kafka()
    else:
        processor.run_batch(Path(args.input_file))


if __name__ == "__main__":
    main()
