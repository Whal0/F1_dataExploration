import fastf1


fastf1.Cache.enable_cache('cache')


race_rounds = list(range(1, 24))
season = 2023

for rnd in race_rounds:
    print(f"Processing {season} round {rnd}...")
    try:
        session = fastf1.get_session(season, rnd, 'R')
        session.load()
        print(f"Cached: {season} round {rnd}")
    except Exception as e:
        print(f"Failed to cache {season} round {rnd}: {e}")