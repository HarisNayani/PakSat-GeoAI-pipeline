from app import demo_data
from paksat.air_quality import summarize_city_pm25


def main() -> None:
    frame = demo_data()
    for city in ["Lahore", "Karachi", "Islamabad"]:
        summary = summarize_city_pm25(frame, city)
        print(f"{city}:")
        print(f"  latest_pm25={summary.latest_pm25}")
        print(f"  trailing_24h_hour_count={summary.trailing_24h_hour_count}")
        print(f"  trailing_24h_mean={summary.trailing_24h_mean}")
        print(f"  latest_timestamp={summary.latest_timestamp}")
        print()


if __name__ == "__main__":
    main()
