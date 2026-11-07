// Geolocation is requested only inside the user's click handler. No IP lookup.
export default function(component) {
  const {parentElement, data, setTriggerValue} = component;
  const root = parentElement;
  const use = root.querySelector('[data-use]');
  const forget = root.querySelector('[data-forget]');
  const remember = root.querySelector('[data-remember]');
  const status = root.querySelector('[data-status]');
  let alive = true;
  const key = 'forecast.exe.preferred-city.v1';
  const emit = value => { if (alive) setTriggerValue('event', {...value, id:crypto.randomUUID()}); };
  const message = value => { status.textContent = value; };
  const storageError = () => message('Browser storage is unavailable. City search still works for this visit.');

  // Restore once per page load, never request permission during restoration.
  if (!window.__forecastLocationRestored) {
    window.__forecastLocationRestored = true;
    try {
      const saved = JSON.parse(localStorage.getItem(key) || 'null');
      if (saved && saved.city) emit({kind:'restore', city:saved.city});
      else emit({kind:'ready'});
    } catch (_) { storageError(); emit({kind:'ready', storage_error:true}); }
  }
  // Only explicit city selections/preferences produce write instructions.
  const write = data?.write;
  if (write && window.__forecastLocationRevision !== write.revision) {
    window.__forecastLocationRevision = write.revision;
    try {
      if (write.city && write.remember) localStorage.setItem(key, JSON.stringify({city:write.city}));
      else localStorage.removeItem(key);
    } catch (_) { storageError(); emit({kind:'storage-error'}); }
  }
  remember.checked = data?.remember !== false;
  use.onclick = () => {
    if (!navigator.geolocation || !window.isSecureContext) {
      message('Location needs HTTPS or localhost. Please use city search.');
      emit({kind:'error', message:'Location is unavailable here. Please use city search.'});
      return;
    }
    use.disabled = true;
    message('Waiting for location permission…');
    try {
      navigator.geolocation.getCurrentPosition(position => {
        if (!alive) return;
        use.disabled = false;
        message('Finding a nearby city…');
        emit({kind:'detected', latitude:Number(position.coords.latitude.toFixed(2)),
              longitude:Number(position.coords.longitude.toFixed(2)), remember:remember.checked});
      }, error => {
        if (!alive) return;
        use.disabled = false;
        const messages = {1:'Location permission was declined. Please search for your city.',
                          2:'Your device could not find a location. Please use city search.',
                          3:'Location timed out. Try again or use city search.'};
        const text = messages[error.code] || 'Location is unavailable. Please use city search.';
        message(text);
        emit({kind:'error', message:text});
      }, {enableHighAccuracy:false, maximumAge:60000, timeout:12000});
    } catch (_) {
      use.disabled = false;
      message('Location is blocked by this browser. Please use city search.');
    }
  };
  remember.onchange = () => emit({kind:'remember', remember:remember.checked});
  forget.onclick = () => {
    let storage_error = false;
    try {localStorage.removeItem(key);} catch (_) {storageError();storage_error=true;}
    emit({kind:'forget',storage_error});
    message('Saved city forgotten.');
  };
  return () => {alive = false; use.onclick = null; forget.onclick = null; remember.onchange = null;};
}
