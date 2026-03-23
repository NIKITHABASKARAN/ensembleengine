import math

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def check_impossible_travel(prev_lat, prev_lon, prev_ts,
                             lat, lon, ts, max_kmh=900.0):
    dist = haversine_km(prev_lat, prev_lon, lat, lon)
    dt_h = (ts - prev_ts) / 3600.0
    if dt_h <= 0:
        return {
            "flagged": True,
            "reason": "timestamp_regression",
            "distance_km": round(dist, 2),
            "velocity_kmh": None
        }
    vel = dist / dt_h
    flagged = vel > max_kmh
    return {
        "flagged": flagged,
        "reason": "impossible_travel" if flagged else "ok",
        "distance_km": round(dist, 2),
        "velocity_kmh": round(vel, 2)
    }
