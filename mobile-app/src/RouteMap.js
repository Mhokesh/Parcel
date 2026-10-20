import React, { useEffect, useRef } from 'react';
import MapView, { Marker, Polyline } from 'react-native-maps';
export default function RouteMap({ hubs, polylines = [], selected = 'all' }) {
  const ref = useRef(null);
  const visible = polylines.filter(p => selected === 'all' || p.route_id === selected);
  const ids = selected === 'all' ? null : new Set(visible.flatMap(p => p.stops));
  const points = hubs.filter(h => !ids || ids.has(h.id));
  const fit = () => { if (points.length) ref.current?.fitToCoordinates(points.map(h => ({ latitude: h.lat, longitude: h.lon })), { edgePadding: { top: 35, bottom: 35, left: 35, right: 35 }, animated: true }); };
  useEffect(() => { const timer = setTimeout(fit, 300); return () => clearTimeout(timer); }, [hubs, polylines, selected]);
  return <MapView ref={ref} style={{ flex: 1 }} userInterfaceStyle="light" onMapReady={fit} initialRegion={{ latitude: 11.0168, longitude: 76.98, latitudeDelta: .14, longitudeDelta: .14 }}>
    {points.map(h => <Marker key={h.id} coordinate={{ latitude: h.lat, longitude: h.lon }} title={h.name} description={h.id === 0 ? 'Central depot' : `${h.demand} kg parcel`} pinColor={h.id === 0 ? '#527661' : '#b98b6b'} />)}
    {visible.map(p => <Polyline key={p.route_id} coordinates={p.polyline.map(([latitude, longitude]) => ({ latitude, longitude }))} strokeColor={p.color} strokeWidth={3} lineDashPattern={[7, 5]} />)}
  </MapView>;
}
