/**
 * AeroPilot Travel Agent — Client Application
 * Handles chat communication, live timeline rendering, Leaflet map route arcs,
 * conflict matrix detection, agent brain inspector, and calendar/PDF export.
 */

const $ = id => document.getElementById(id);

// App State
let sessionId = crypto.randomUUID();
let messageCount = 0;
let currentBookings = [];
let currentConflicts = [];
let currentItinerary = null;
let currentFilter = 'all';
let leafletMap = null;
let mapMarkersGroup = null;
let mapLinesGroup = null;

// Built-in Geocoding Dictionary for Clean Map Rendering
const CITY_COORDS = {
  'islamabad': [33.6844, 73.0479],
  'isb': [33.6844, 73.0479],
  'istanbul': [41.0082, 28.9784],
  'ist': [41.0082, 28.9784],
  'saw': [40.8986, 29.3092],
  'london': [51.5074, -0.1278],
  'lhr': [51.4700, -0.4543],
  'lgw': [51.1537, -0.1821],
  'paris': [48.8566, 2.3522],
  'cdg': [49.0097, 2.5479],
  'ory': [48.7262, 2.3652],
  'rome': [41.9028, 12.4964],
  'fco': [41.8003, 12.2389],
  'new york': [40.7128, -74.0060],
  'jfk': [40.6413, -73.7781],
  'lax': [33.9416, -118.4085],
  'los angeles': [34.0522, -118.2437],
  'atlanta': [33.7490, -84.3880],
  'atl': [33.6407, -84.4277],
  'amsterdam': [52.3676, 4.9041],
  'ams': [52.3105, 4.7683],
  'frankfurt': [50.1109, 8.6821],
  'fra': [50.0379, 8.5622],
  'dubai': [25.2048, 55.2708],
  'dxb': [25.2532, 55.3657],
  'tokyo': [35.6762, 139.6503],
  'hnd': [35.5494, 139.7798],
  'nrt': [35.7720, 140.3929],
  'barcelona': [41.3851, 2.1734],
  'bcn': [41.2974, 2.0833],
  'berlin': [52.5200, 13.4050],
  'ber': [52.3667, 13.5033],
  'zurich': [47.3769, 8.5417],
  'zrh': [47.4582, 8.5555],
  'cairo': [30.0444, 31.2357],
  'lahore': [31.5204, 74.3587],
  'karachi': [24.8607, 67.0011],
  'sydney': [-33.8688, 151.2093],
  'singapore': [1.3521, 103.8198],
  'sin': [1.3644, 103.9915],
  'san francisco': [37.7749, -122.4194],
  'sfo': [37.6213, -122.3790]
};

// ─── Initialization ─────────────────────────────────────────────────
async function init() {
  bindEvents();
  initLeafletMap();
  await loadModels();
}

function bindEvents() {
  // Chat submit
  $('chat').addEventListener('submit', handleChatSubmit);
  
  // Reset chat
  $('reset').addEventListener('click', handleReset);

  // Agent Brain Drawer Toggle
  $('toggle-brain-btn').addEventListener('click', toggleBrainDrawer);
  $('close-brain-btn').addEventListener('click', toggleBrainDrawer);

  // Tab switching
  $('tab-timeline-btn').addEventListener('click', () => switchTab('timeline'));
  $('tab-map-btn').addEventListener('click', () => switchTab('map'));
  $('tab-conflicts-btn').addEventListener('click', () => switchTab('conflicts'));

  // Conflict Jump Button
  $('conflict-jump-btn').addEventListener('click', () => switchTab('conflicts'));

  // Filter pills
  document.querySelectorAll('.filter-pill').forEach(pill => {
    pill.addEventListener('click', e => {
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      e.target.classList.add('active');
      currentFilter = e.target.dataset.filter;
      renderTimeline();
    });
  });

  // Context Attachment Toggle
  $('toggle-context-btn').addEventListener('click', () => {
    $('context-drawer').classList.toggle('hidden');
    if (!$('context-drawer').classList.contains('hidden')) {
      $('external').focus();
    }
  });

  $('close-context-btn').addEventListener('click', () => {
    $('external').value = '';
    $('context-drawer').classList.add('hidden');
  });

  // Quick Scenarios Chips
  document.querySelectorAll('.scenario-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      $('task').value = btn.dataset.task || '';
      $('external').value = btn.dataset.note || '';
      if (btn.dataset.note) {
        $('context-drawer').classList.remove('hidden');
      } else {
        $('context-drawer').classList.add('hidden');
      }
      $('task').focus();
    });
  });

  // Export Buttons
  $('export-btn').addEventListener('click', e => {
    e.stopPropagation();
    $('export-menu').classList.toggle('show');
  });
  document.addEventListener('click', () => $('export-menu').classList.remove('show'));

  $('export-pdf-btn').addEventListener('click', handleExportPDF);
  $('export-ics-btn').addEventListener('click', handleExportICS);
  $('copy-summary-btn').addEventListener('click', handleCopyMarkdown);

  // Textarea auto-resize & keyboard shortcuts
  const taskInput = $('task');
  taskInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      $('chat').requestSubmit();
    }
  });
}

// ─── Model Loading ──────────────────────────────────────────────────
async function loadModels() {
  try {
    const res = await fetch('/models');
    if (!res.ok) throw new Error('Failed to load models');
    const data = await res.json();
    const select = $('model');
    select.innerHTML = '';
    (data.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Model fetch error:', err);
    $('model').innerHTML = '<option value="unconfigured">Default Model</option>';
  }
}

// ─── Chat Handling ──────────────────────────────────────────────────
async function handleChatSubmit(e) {
  e.preventDefault();
  const task = $('task').value.trim();
  if (!task) return;

  const note = $('external').value.trim();
  
  // Add User Message to Chat Stream
  appendMessage('user', task);
  $('task').value = '';
  
  // Set UI to Running State
  setRunningState(true);
  const typingEl = appendTypingIndicator();

  try {
    const payload = {
      session_id: sessionId,
      task: task,
      model: $('model').value,
      external_context: note ? [{ source: 'user-note', content: note, trust: 'untrusted' }] : []
    };

    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    typingEl.remove();

    if (!res.ok) {
      throw new Error(data.detail || 'Server encountered an error processing your request.');
    }

    // Append Agent Response
    appendMessage('agent', data.final_response, data.status, data.tool_calls);
    
    // Process State Changes & Canvas Updates
    messageCount += 2;
    $('memory-counter').textContent = `${messageCount} messages exchanged · Server retains up to 12`;

    processArenaResponse(data);

  } catch (err) {
    typingEl.remove();
    appendMessage('agent', `⚠️ Error: ${err.message}`, 'failed');
    updateStatusChip('alert');
  } finally {
    setRunningState(false);
  }
}

function setRunningState(running) {
  $('send').disabled = running;
  $('reset').disabled = running;
  updateStatusChip(running ? 'running' : 'idle');
}

function updateStatusChip(status) {
  const chip = $('agent-status-tag');
  chip.className = `status-chip ${status}`;
  chip.textContent = status.toUpperCase();
}

// ─── Message Rendering ──────────────────────────────────────────────
function appendMessage(role, text, status = null, toolCalls = []) {
  const article = document.createElement('article');
  article.className = role;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar-wrap';
  avatar.innerHTML = `<div class="msg-avatar">${
    role === 'user' 
      ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
      : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4v1a4 4 0 0 0-4 4v2a4 4 0 0 0 4 4v1a4 4 0 0 0 4 4h0a4 4 0 0 0 4-4v-1a4 4 0 0 0 4-4v-2a4 4 0 0 0-4-4V6a4 4 0 0 0-4-4z"/></svg>'
  }</div>`;

  const body = document.createElement('div');
  body.className = 'msg-body';

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const senderLabel = role === 'user' ? 'You' : 'AeroPilot AI';

  let toolBadgesHtml = '';
  if (toolCalls && toolCalls.length > 0) {
    toolBadgesHtml = `<div class="msg-tool-badges">${
      toolCalls.map(tc => `<span class="tool-pill ${tc.outcome}">⚙️ ${tc.tool} (${tc.outcome})</span>`).join('')
    }</div>`;
  }

  // Simple clean formatting for markdown-style bullet points and bolding
  const formattedText = formatChatMessage(text);

  body.innerHTML = `
    <div class="msg-header">
      <span class="sender-name">${senderLabel}</span>
      <span class="timestamp">${timeStr}</span>
    </div>
    <div class="msg-content">${formattedText}</div>
    ${toolBadgesHtml}
  `;

  article.appendChild(avatar);
  article.appendChild(body);
  $('messages').appendChild(article);
  article.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function formatChatMessage(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n- (.*?)/g, '<br>• $1')
    .replace(/\n/g, '<br>');
}

function appendTypingIndicator() {
  const article = document.createElement('article');
  article.className = 'agent';
  article.innerHTML = `
    <div class="msg-avatar-wrap">
      <div class="msg-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4v1a4 4 0 0 0-4 4v2a4 4 0 0 0 4 4v1a4 4 0 0 0 4 4h0a4 4 0 0 0 4-4v-1a4 4 0 0 0 4-4v-2a4 4 0 0 0-4-4V6a4 4 0 0 0-4-4z"/></svg>
      </div>
    </div>
    <div class="msg-body">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>
  `;
  $('messages').appendChild(article);
  article.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return article;
}

// ─── Arena Response Processing & Data Extraction ────────────────────
function processArenaResponse(data) {
  // Update Agent Brain Metrics
  $('nav-step-badge').textContent = `${data.steps}/6`;
  $('metric-status').textContent = data.status;
  $('metric-steps').textContent = `${data.steps} / 6`;
  $('metric-latency').textContent = `${Math.round(data.metrics?.latency_ms || 0)} ms`;
  $('metric-tokens').textContent = `${data.metrics?.input_tokens ?? '—'} / ${data.metrics?.output_tokens ?? '—'}`;
  
  // Render Step Timeline in Brain Inspector
  renderBrainInspector(data);

  // Extract Bookings, Conflicts, and Itinerary from Events
  let stopEvent = data.events?.find(e => e.event === 'agent_stop');
  let toolEvents = data.events?.filter(e => e.event === 'tool_execution') || [];

  if (stopEvent) {
    if (stopEvent.bookings && stopEvent.bookings.length > 0) {
      currentBookings = stopEvent.bookings;
    }
    currentConflicts = stopEvent.conflicts || [];
    if (stopEvent.itinerary) {
      currentItinerary = stopEvent.itinerary;
    }
  }

  // If stopEvent didn't contain bookings, check tool execution data
  if (currentBookings.length === 0) {
    const recordToolEvent = toolEvents.find(e => e.tool === 'record_bookings' && e.data?.bookings);
    if (recordToolEvent) {
      currentBookings = recordToolEvent.data.bookings;
    }
  }

  const buildToolEvent = toolEvents.find(e => e.tool === 'build_itinerary' && e.data?.itinerary);
  if (buildToolEvent) {
    currentItinerary = buildToolEvent.data.itinerary;
    if (buildToolEvent.data.conflicts) {
      currentConflicts = buildToolEvent.data.conflicts;
    }
  }

  // Update Canvas Header Badges
  $('canvas-badge-bookings').textContent = `${currentBookings.length} Booking${currentBookings.length === 1 ? '' : 's'}`;
  if (currentConflicts.length > 0) {
    $('canvas-badge-conflicts').textContent = `${currentConflicts.length} Conflict${currentConflicts.length === 1 ? '' : 's'}`;
    $('canvas-badge-conflicts').classList.remove('hidden');
    $('conflict-count-badge').textContent = currentConflicts.length;
    $('conflict-count-badge').classList.remove('hidden');
    showConflictBanner(true);
  } else {
    $('canvas-badge-conflicts').classList.add('hidden');
    $('conflict-count-badge').classList.add('hidden');
    showConflictBanner(false);
  }

  if (currentItinerary?.title) {
    $('trip-title').textContent = currentItinerary.title;
  } else if (currentBookings.length > 0) {
    $('trip-title').textContent = `Journey to ${currentBookings[0].location || 'Destination'}`;
  }

  if (currentBookings.length > 0) {
    const sorted = [...currentBookings].sort((a,b) => new Date(a.start_datetime) - new Date(b.start_datetime));
    const firstDate = new Date(sorted[0].start_datetime).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
    const lastDate = new Date(sorted[sorted.length-1].end_datetime).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
    $('trip-dates').textContent = `${firstDate} — ${lastDate}`;
  }

  // Render Sub-Views
  renderTimeline();
  renderConflictsView();
  updateLeafletMap();
}

// ─── Timeline Rendering (Pastel Essence Cards) ──────────────────────
function renderTimeline() {
  const container = $('timeline-schedule');
  const emptyState = $('timeline-empty');

  if (currentBookings.length === 0) {
    emptyState.classList.remove('hidden');
    container.innerHTML = '';
    return;
  }

  emptyState.classList.add('hidden');
  container.innerHTML = '';

  // Filter bookings if requested
  const filtered = currentBookings.filter(b => {
    if (currentFilter === 'all') return true;
    return b.category === currentFilter;
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div class="canvas-empty-state"><p>No ${currentFilter} bookings found for this trip.</p></div>`;
    return;
  }

  // Group by UTC Day
  const grouped = {};
  filtered.sort((a, b) => new Date(a.start_datetime) - new Date(b.start_datetime));
  
  filtered.forEach(b => {
    const dayKey = b.start_datetime.split('T')[0];
    if (!grouped[dayKey]) grouped[dayKey] = [];
    grouped[dayKey].push(b);
  });

  let dayIndex = 1;
  for (const [dateStr, items] of Object.entries(grouped)) {
    const dayBlock = document.createElement('div');
    dayBlock.className = 'timeline-day-block';

    const dateObj = new Date(dateStr + 'T00:00:00Z');
    const dayFormatted = dateObj.toLocaleDateString([], { weekday: 'long', month: 'short', day: 'numeric', year: 'numeric' });

    dayBlock.innerHTML = `
      <div class="day-header">
        <span class="day-pill">Day ${dayIndex}</span>
        <span class="day-date-str">${dayFormatted}</span>
        <span class="day-items-count">${items.length} item${items.length === 1 ? '' : 's'}</span>
      </div>
      <div class="day-cards-list"></div>
    `;

    const cardsList = dayBlock.querySelector('.day-cards-list');
    items.forEach(item => {
      cardsList.appendChild(createBookingCard(item));
    });

    container.appendChild(dayBlock);
    dayIndex++;
  }
}

function createBookingCard(item) {
  const card = document.createElement('div');
  const isConflict = currentConflicts.some(c => 
    c.booking_1?.confirmation_code === item.confirmation_code || 
    c.booking_2?.confirmation_code === item.confirmation_code
  );

  card.className = `booking-card ${item.category}-card ${isConflict ? 'has-conflict' : ''}`;

  if (item.category === 'transport') {
    // Flight / Train Boarding Pass Card
    const route = extractFlightRoute(item.title, item.location);

    card.innerHTML = `
      <div class="card-top-bar">
        <span class="category-badge transport">✈️ ${item.type || 'Flight'}</span>
        <span class="conf-code-pill">${item.confirmation_code}</span>
      </div>
      <div class="flight-route-display">
        <div class="iata-block">
          <div class="iata-code">${route.fromIATA}</div>
          <div class="iata-city">${route.fromCity}</div>
        </div>
        <div class="flight-connector">
          <div class="flight-line"></div>
          <div class="flight-icon-center">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.3c.4-.2.6-.6.5-1.1z"/></svg>
          </div>
        </div>
        <div class="iata-block">
          <div class="iata-code">${route.toIATA}</div>
          <div class="iata-city">${route.toCity}</div>
        </div>
      </div>
      <div class="card-times-bar">
        <span>Dep: <span class="time-val">${item.start_text}</span></span>
        <span>Arr: <span class="time-val">${item.end_text}</span></span>
      </div>
      ${isConflict ? `
        <div class="card-conflict-warning">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span>Conflict detected for this travel interval</span>
        </div>` : ''}
    `;
  } else if (item.category === 'accommodation') {
    // Hotel / Stay Card
    card.innerHTML = `
      <div class="card-top-bar">
        <span class="category-badge accommodation">🏨 ${item.type || 'Hotel'}</span>
        <span class="conf-code-pill">${item.confirmation_code}</span>
      </div>
      <h3 class="card-main-title">${item.title}</h3>
      <div class="card-location-row">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
        <span>${item.location}</span>
      </div>
      <div class="card-times-bar">
        <span>Check-in: <span class="time-val">${item.start_text}</span></span>
        <span>Check-out: <span class="time-val">${item.end_text}</span></span>
      </div>
      ${isConflict ? `
        <div class="card-conflict-warning">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span>Overlapping hotel reservation detected</span>
        </div>` : ''}
    `;
  } else {
    // Activity / Tour Card
    card.innerHTML = `
      <div class="card-top-bar">
        <span class="category-badge activity">🎟️ ${item.type || 'Activity'}</span>
        <span class="conf-code-pill">${item.confirmation_code}</span>
      </div>
      <h3 class="card-main-title">${item.title}</h3>
      <div class="card-location-row">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
        <span>${item.location}</span>
      </div>
      <div class="card-times-bar">
        <span>Start: <span class="time-val">${item.start_text}</span></span>
        <span>End: <span class="time-val">${item.end_text}</span></span>
      </div>
      ${isConflict ? `
        <div class="card-conflict-warning">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span>Activity schedule collision</span>
        </div>` : ''}
    `;
  }

  return card;
}

const IATA_MAP = {
  'islamabad': 'ISB', 'isb': 'ISB',
  'istanbul': 'IST', 'ist': 'IST', 'saw': 'SAW',
  'london': 'LHR', 'lhr': 'LHR', 'lgw': 'LGW',
  'paris': 'CDG', 'cdg': 'CDG', 'ory': 'ORY',
  'rome': 'FCO', 'fco': 'FCO',
  'new york': 'JFK', 'jfk': 'JFK', 'nyc': 'JFK',
  'los angeles': 'LAX', 'lax': 'LAX',
  'atlanta': 'ATL', 'atl': 'ATL',
  'amsterdam': 'AMS', 'ams': 'AMS',
  'frankfurt': 'FRA', 'fra': 'FRA',
  'dubai': 'DXB', 'dxb': 'DXB',
  'tokyo': 'HND', 'hnd': 'HND', 'nrt': 'NRT',
  'barcelona': 'BCN', 'bcn': 'BCN',
  'berlin': 'BER', 'ber': 'BER',
  'zurich': 'ZRH', 'zrh': 'ZRH',
  'lahore': 'LHE', 'karachi': 'KHI',
  'singapore': 'SIN', 'sin': 'SIN',
  'san francisco': 'SFO', 'sfo': 'SFO'
};

function getIATACode(cityName) {
  if (!cityName) return 'DEP';
  const clean = cityName.toLowerCase().trim();
  for (const [key, code] of Object.entries(IATA_MAP)) {
    if (clean.includes(key)) return code;
  }
  return cityName.replace(/[^A-Za-z]/g, '').slice(0, 3).toUpperCase() || 'DEP';
}

function extractFlightRoute(title, defaultLoc) {
  const combined = `${title || ''} ${defaultLoc || ''}`.trim();
  if (!combined) return { fromCity: 'Departure', toCity: 'Arrival', fromIATA: 'DEP', toIATA: 'ARR' };
  
  // Try pattern: "from X to Y" or "X to Y"
  const match = combined.match(/(?:from\s+)?([A-Za-z]+)\s+to\s+([A-Za-z]+)/i);
  if (match) {
    const fromCity = match[1].trim();
    const toCity = match[2].trim();
    return {
      fromCity,
      toCity,
      fromIATA: getIATACode(fromCity),
      toIATA: getIATACode(toCity)
    };
  }

  // Fallback: check location
  const toCity = defaultLoc || 'Destination';
  return {
    fromCity: 'Origin',
    toCity,
    fromIATA: 'DEP',
    toIATA: getIATACode(toCity)
  };
}

// ─── Conflict Matrix Rendering ──────────────────────────────────────
function showConflictBanner(show) {
  const banner = $('conflict-alert-banner');
  if (show && currentConflicts.length > 0) {
    banner.classList.remove('hidden');
    $('conflict-banner-title').textContent = `${currentConflicts.length} Scheduling Collision${currentConflicts.length === 1 ? '' : 's'} Flagged`;
    $('conflict-banner-desc').textContent = currentConflicts[0].description || 'Interval collision detected.';
  } else {
    banner.classList.add('hidden');
  }
}

function renderConflictsView() {
  const emptyState = $('conflicts-empty');
  const list = $('conflicts-list');

  if (currentConflicts.length === 0) {
    emptyState.classList.remove('hidden');
    list.innerHTML = '';
    return;
  }

  emptyState.classList.add('hidden');
  list.innerHTML = '';

  currentConflicts.forEach(conf => {
    const card = document.createElement('div');
    card.className = 'conflict-detail-card';

    const b1 = conf.booking_1 || {};
    const b2 = conf.booking_2 || {};
    const ovStart = conf.overlap_start ? new Date(conf.overlap_start).toUTCString() : '—';
    const ovEnd = conf.overlap_end ? new Date(conf.overlap_end).toUTCString() : '—';

    card.innerHTML = `
      <div class="conflict-card-header">
        <span class="conflict-badge">⚠️ ${conf.category || 'Category'} Collision</span>
        <span class="overlap-window">${ovStart} → ${ovEnd}</span>
      </div>
      <p class="conflict-card-desc">${conf.description}</p>
      <div class="colliding-items-grid">
        <div class="colliding-box">
          <strong>Booking 1 (${b1.confirmation_code || '—'})</strong>
          <p>${b1.title || 'Untitled'}</p>
          <small>${b1.start_text || ''} to ${b1.end_text || ''}</small>
        </div>
        <div class="colliding-box">
          <strong>Booking 2 (${b2.confirmation_code || '—'})</strong>
          <p>${b2.title || 'Untitled'}</p>
          <small>${b2.start_text || ''} to ${b2.end_text || ''}</small>
        </div>
      </div>
    `;

    list.appendChild(card);
  });
}

// ─── Leaflet Map Integration ────────────────────────────────────────
function initLeafletMap() {
  if (leafletMap) return;

  const mapEl = $('map-container');
  if (!mapEl || typeof L === 'undefined') return;

  // Center around Europe/Mediterranean as default view
  leafletMap = L.map('map-container', {
    zoomControl: true,
    scrollWheelZoom: true
  }).setView([45.0, 15.0], 4);

  // Esri World Light Gray Canvas: Premium, clean, pastel light map tiles without watermarks
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    maxZoom: 16
  }).addTo(leafletMap);

  mapMarkersGroup = L.layerGroup().addTo(leafletMap);
  mapLinesGroup = L.layerGroup().addTo(leafletMap);
}

function updateLeafletMap() {
  if (!leafletMap) return;

  mapMarkersGroup.clearLayers();
  mapLinesGroup.clearLayers();

  const bounds = [];

  currentBookings.forEach(b => {
    const loc = (b.location || b.title || '').toLowerCase();
    let coords = resolveCoords(loc);

    if (b.category === 'transport') {
      const route = extractFlightRoute(b.title, b.location);
      const c1 = resolveCoords(route.fromCity);
      const c2 = resolveCoords(route.toCity);

      if (c1 && c2) {
        bounds.push(c1, c2);
        
        // Add markers
        addMapMarker(c1, `🛫 ${route.fromCity} (${route.fromIATA})`, '#3874cb');
        addMapMarker(c2, `🛬 ${route.toCity} (${route.toIATA})`, '#3874cb');

        // Draw curved flight arc between coordinates
        drawCurvedFlightArc(c1, c2, `${route.fromIATA} → ${route.toIATA}`);
      } else if (coords) {
        bounds.push(coords);
        addMapMarker(coords, b.title, '#3874cb');
      }
    } else {
      if (coords) {
        bounds.push(coords);
        const color = b.category === 'accommodation' ? '#1e824c' : '#7c3aed';
        addMapMarker(coords, `${b.category === 'accommodation' ? '🏨' : '🎟️'} ${b.title}`, color);
      }
    }
  });

  if (bounds.length > 0) {
    leafletMap.fitBounds(bounds, { padding: [40, 40], maxZoom: 8 });
  }
}

function resolveCoords(str) {
  if (!str) return null;
  const clean = str.toLowerCase();
  for (const [key, coords] of Object.entries(CITY_COORDS)) {
    if (clean.includes(key)) {
      return coords;
    }
  }
  // Fallback random mild offset near Europe/Middle East for realistic visualization
  return [41.9 + (Math.random() - 0.5) * 8, 12.5 + (Math.random() - 0.5) * 14];
}

function addMapMarker(coords, title, color) {
  const iconHtml = `<div class="custom-map-marker" style="border-color:${color};color:${color};">📍</div>`;
  const icon = L.divIcon({ html: iconHtml, className: 'map-div-icon', iconSize: [30, 30] });

  L.marker(coords, { icon })
    .bindPopup(`<strong>${title}</strong>`)
    .addTo(mapMarkersGroup);
}

function drawCurvedFlightArc(c1, c2, title) {
  // Approximate a smooth curved trajectory by computing midpoint with elevation offset
  const midLat = (c1[0] + c2[0]) / 2 + 2.5;
  const midLng = (c1[1] + c2[1]) / 2;

  const latlngs = [c1, [midLat, midLng], c2];
  
  const curve = L.polyline(latlngs, {
    color: '#3874cb',
    weight: 3,
    opacity: 0.8,
    dashArray: '6, 8',
    smoothFactor: 1
  }).bindPopup(`<strong>Flight Route:</strong> ${title}`).addTo(mapLinesGroup);
}

// ─── Agent Brain Inspector Rendering ────────────────────────────────
function renderBrainInspector(data) {
  // Trace Feed
  const traceFeed = $('brain-steps-trace');
  traceFeed.innerHTML = '';

  const events = data.events || [];
  if (events.length === 0) {
    traceFeed.innerHTML = '<p class="placeholder-text">No events logged for this turn.</p>';
  } else {
    events.forEach(ev => {
      const stepRow = document.createElement('div');
      stepRow.className = 'trace-step-item';

      let stepNum = ev.step || 1;
      let title = ev.event || 'Action';
      let latency = ev.latency_ms ? `${Math.round(ev.latency_ms)}ms` : '';
      let desc = '';

      if (ev.event === 'model_call') {
        title = 'LLM Inference';
        desc = `Tokens: In=${ev.input_tokens || '—'}, Out=${ev.output_tokens || '—'} · Finish=${ev.finish_reason || 'stop'}`;
      } else if (ev.event === 'decision_validated') {
        title = `Decision: ${ev.status}`;
        desc = ev.action ? `Selected Tool Action: ${ev.action}` : 'Terminal decision verified';
      } else if (ev.event === 'tool_execution') {
        title = `Tool Execution: ${ev.tool}`;
        desc = `Outcome: ${ev.outcome}`;
      } else if (ev.event === 'repair_attempt') {
        title = 'Self-Repair Activated';
        desc = 'One-shot prompt repair loop triggered';
      } else if (ev.event === 'agent_stop') {
        title = `Agent Finished (${ev.status})`;
        desc = `Stop reason: ${ev.reason}`;
      }

      stepRow.innerHTML = `
        <div class="step-number-bubble">${stepNum}</div>
        <div class="step-details">
          <div class="step-title-row">
            <span class="step-name">${title}</span>
            <span class="step-latency">${latency}</span>
          </div>
          <div class="step-desc">${desc}</div>
        </div>
      `;
      traceFeed.appendChild(stepRow);
    });
  }

  // Tool Calls Feed
  const toolFeed = $('brain-tool-calls');
  toolFeed.innerHTML = '';
  if (!data.tool_calls || data.tool_calls.length === 0) {
    toolFeed.innerHTML = '<p class="placeholder-text">No tool gateway executions in this run.</p>';
  } else {
    data.tool_calls.forEach(tc => {
      const row = document.createElement('div');
      row.className = 'tool-call-row';
      row.innerHTML = `
        <span class="tool-call-name">gateway::${tc.tool}</span>
        <span class="tool-outcome-pill ${tc.outcome}">${tc.outcome} (${Math.round(tc.latency_ms)}ms)</span>
      `;
      toolFeed.appendChild(row);
    });
  }

  // Raw JSON
  $('observations').textContent = JSON.stringify(data, null, 2);
}

function toggleBrainDrawer() {
  $('brain-drawer').classList.toggle('open');
}

// ─── Tab Switching ──────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-pill').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.canvas-view').forEach(view => view.classList.remove('active'));

  if (tabId === 'timeline') {
    $('tab-timeline-btn').classList.add('active');
    $('view-timeline').classList.add('active');
  } else if (tabId === 'map') {
    $('tab-map-btn').classList.add('active');
    $('view-map').classList.add('active');
    if (leafletMap) {
      setTimeout(() => leafletMap.invalidateSize(), 150);
    }
  } else if (tabId === 'conflicts') {
    $('tab-conflicts-btn').classList.add('active');
    $('view-conflicts').classList.add('active');
  }
}

// ─── Reset Session ──────────────────────────────────────────────────
async function handleReset() {
  try {
    await fetch(`/chat/${sessionId}`, { method: 'DELETE' });
  } catch (err) {
    console.warn('Reset endpoint call:', err);
  }

  sessionId = crypto.randomUUID();
  messageCount = 0;
  currentBookings = [];
  currentConflicts = [];
  currentItinerary = null;

  $('messages').innerHTML = `
    <article class="agent greeting-card">
      <div class="msg-avatar-wrap">
        <div class="msg-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4v1a4 4 0 0 0-4 4v2a4 4 0 0 0 4 4v1a4 4 0 0 0 4 4h0a4 4 0 0 0 4-4v-1a4 4 0 0 0 4-4v-2a4 4 0 0 0-4-4V6a4 4 0 0 0-4-4z"/></svg>
        </div>
      </div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="sender-name">AeroPilot AI</span>
          <span class="timestamp">Just now</span>
        </div>
        <p>New trip session started. Paste your flights, hotels, or tickets to build your itinerary!</p>
      </div>
    </article>
  `;

  $('memory-counter').textContent = '0 messages exchanged · Isolated sandbox';
  $('task').value = '';
  $('external').value = '';
  $('context-drawer').classList.add('hidden');
  $('trip-title').textContent = 'Voyage Overview';
  $('trip-dates').textContent = 'No active bookings recorded yet';
  $('canvas-badge-bookings').textContent = '0 Bookings';
  $('canvas-badge-conflicts').classList.add('hidden');
  $('conflict-count-badge').classList.add('hidden');
  showConflictBanner(false);
  updateStatusChip('idle');

  renderTimeline();
  renderConflictsView();
  if (leafletMap) {
    mapMarkersGroup.clearLayers();
    mapLinesGroup.clearLayers();
  }
}

// ─── Export Features (PDF, iCal, Markdown) ───────────────────────────
function handleExportPDF() {
  window.print();
}

function handleExportICS() {
  if (currentBookings.length === 0) {
    alert('No bookings available to export. Please record some flights or stays first!');
    return;
  }

  let icsContent = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//AeroPilot Travel Agent//EN',
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH'
  ];

  currentBookings.forEach(b => {
    const s = formatICalDate(b.start_datetime);
    const e = formatICalDate(b.end_datetime);
    const uid = `${b.confirmation_code}-${Date.now()}@aeropilot.local`;

    icsContent.push(
      'BEGIN:VEVENT',
      `UID:${uid}`,
      `DTSTAMP:${formatICalDate(new Date().toISOString())}`,
      `DTSTART:${s}`,
      `DTEND:${e}`,
      `SUMMARY:${escapeICal(b.title)}`,
      `LOCATION:${escapeICal(b.location || '')}`,
      `DESCRIPTION:Confirmation Code: ${b.confirmation_code}\\nCategory: ${b.category}`,
      'END:VEVENT'
    );
  });

  icsContent.push('END:VCALENDAR');

  const blob = new Blob([icsContent.join('\r\n')], { type: 'text/calendar;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `itinerary-${new Date().toISOString().slice(0, 10)}.ics`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function formatICalDate(isoString) {
  const d = new Date(isoString);
  return d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
}

function escapeICal(str) {
  return (str || '').replace(/[\\;,]/g, '\\$&');
}

function handleCopyMarkdown() {
  if (currentBookings.length === 0) {
    alert('No itinerary schedule to copy yet.');
    return;
  }

  let md = `# Trip Itinerary: ${$('trip-title').textContent}\n\n`;
  currentBookings.forEach(b => {
    md += `- **[${b.category.toUpperCase()}]** ${b.title}\n`;
    md += `  - Times: ${b.start_text} → ${b.end_text}\n`;
    md += `  - Confirmation: \`${b.confirmation_code}\`\n`;
    md += `  - Location: ${b.location}\n\n`;
  });

  if (currentConflicts.length > 0) {
    md += `### ⚠️ Detected Conflicts:\n`;
    currentConflicts.forEach(c => {
      md += `- ${c.description}\n`;
    });
  }

  navigator.clipboard.writeText(md).then(() => {
    alert('✅ Itinerary schedule copied to clipboard in Markdown format!');
  });
}

// Start application
window.addEventListener('DOMContentLoaded', init);
