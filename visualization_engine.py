import json
import logging
from typing import List, Dict, Any, Tuple
logger = logging.getLogger(__name__)

class VisualizationEngine:
    def __init__(self, llm_client=None):
        """
        Initializes the visualization engine.
        
        Args:
            llm_client: The existing BedrockLLMClient instance to handle field mapping selection 
                         if programmatic analysis becomes ambiguous.
        """
        self.llm_client = llm_client
        
    def generate_chart_html(self, data: List[Dict[str, Any]], user_prompt: str) -> Dict[str, Any]:
        """
        Main entry point. Analyzes the dataset fields, selects appropriate X/Y charting axes,
        and compiles the structural data into a fully executable dynamic HTML/CSS template snippet.
        """
        if not data:
            return {
                "type": "text",
                "display_value": "No data available to plot a graph."
            }

        # Step 1: Extract all available field keys from the first row record
        available_keys = list(data[0].keys())
        
        # Step 2: Determine X and Y axes using our hybrid approach
        x_axis, y_axis = self._determine_chart_axes(data, available_keys, user_prompt)
        
        if not x_axis or not y_axis:
            return {
                "type": "text",
                "display_value": f"Could not determine clean numeric variables to chart automatically. Available fields: {', '.join(available_keys)}"
            }

        # Step 3: Extract and structure the raw column value arrays out for the chart configuration
        x_labels = [str(row.get(x_axis, "")) for row in data]
        y_values = []
        for row in data:
            val = row.get(y_axis, 0)
            # Ensure numbers are native Python floats or ints
            y_values.append(float(val) if val is not None else 0.0)

        # Step 4: Choose chart type based on field attributes (e.g., date-based line vs object bar)
        chart_type = "line" if "date" in x_axis.lower() or "time" in x_axis.lower() else "bar"
        chart_title = f"{y_axis.replace('_', ' ').title()} grouped by {x_axis.replace('_', ' ').title()}"

        # Step 5: Compile into a fully embedded web template block
        html_snippet = self._compile_chart_js_template(
            chart_type=chart_type,
            chart_title=chart_title,
            x_labels=x_labels,
            y_values=y_values,
            y_label=y_axis.replace('_', ' ').title()
        )

        return {
            "type": "rich_media",
            "display_value": f"Graph rendered successfully showing: {chart_title}.",
            "html_content": html_snippet
        }

    def _determine_chart_axes(self, data: List[Dict[str, Any]], keys: List[str], user_prompt: str) -> Tuple[str, str]:
        """
        Hybrid selection strategy: runs programmatic data-type scanning first, 
        and falls back to a fast, low-overhead LLM choice if fields are ambiguous.
        """
        numeric_candidates = []
        label_candidates = []

        # Analyze data types from the first record entry to detect viable variables
        first_row = data[0]
        for key in keys:
            val = first_row[key]
            # Categorize numeric fields (excluding explicit primary IDs)
            if isinstance(val, (int, float)) and not key.lower().endswith(('_id', 'vid')):
                numeric_candidates.append(key)
            elif isinstance(val, str) or key.lower().endswith(('_id', 'vid', 'date', 'number')):
                label_candidates.append(key)

        # Programmatic Safe Path: If it's a completely clear 1-to-1 matching scenario
        if len(numeric_candidates) == 1 and len(label_candidates) >= 1:
            # Pick the most prominent label key available
            preferred_labels = ['license_plate_number', 'geofence_name', 'date', 'vehicle_id', 'vid']
            selected_x = next((l for l in preferred_labels if l in label_candidates), label_candidates[0])
            return selected_x, numeric_candidates[0]

        # LLM Backup Path: If multiple metric options exist, use a quick metadata-only prompt
        if self.llm_client and len(numeric_candidates) > 1:
            logger.info("[VISUALIZATION] Multiple numeric metrics detected. Invoking fast LLM field resolver.")
            return self._invoke_llm_field_resolver(keys, user_prompt)

        # Default fallback match if no automated AI layer exists
        x_axis = label_candidates[0] if label_candidates else None
        y_axis = numeric_candidates[0] if numeric_candidates else None
        return x_axis, y_axis

    def _invoke_llm_field_resolver(self, keys: List[str], user_prompt: str) -> Tuple[str, str]:
        """
        Asks Bedrock via an isolated call to map keys to axes based purely on user phrasing intent.
        """
        resolver_prompt = f"""You are a mapping specialist for data visualization charts.
Given a list of available keys from a database payload and the user's charting instruction, output exactly two keys: one for the X-axis (labels) and one for the Y-axis (numeric data value measurements).

Available Data Keys: {keys}
User Visualization Intent: "{user_prompt}"

CRITICAL RULES:
1. Output ONLY a valid JSON object matching this structure: {{"x": "selected_key", "y": "selected_key"}}
2. Do not include markdown tags, introduction statements, or thinking logs.

Target JSON Object:"""
        try:
            # Call your low-latency Bedrock method
            response = self.llm_client.condense_history_query(resolver_prompt)
            parsed = json.loads(response.strip())
            return parsed.get("x"), parsed.get("y")
        except Exception as e:
            logger.error(f"Failed parsing LLM field resolver selection layout: {e}")
            return None, None

    def _compile_chart_js_template(self, chart_type: str, chart_title: str, x_labels: List[str], y_values: List[float], y_label: str) -> str:
        """
        Assembles responsive CSS structures, standard HTML components, and a CDN script link 
        to seamlessly populate a beautiful Chart.js widget inside a single render string.
        """
        # Safely dump python structures directly to inline JSON safe syntax strings
        json_x_labels = json.dumps(x_labels)
        json_y_values = json.dumps(y_values)
        
        # Generate a unique chart ID to avoid DOM collisions
        import time
        import random
        chart_id = f"fleetChart_{int(time.time())}_{random.randint(1000, 9999)}"

        # Compute dynamic styles based on default type
        bar_style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-color: #667eea;" if chart_type == "bar" else "background: #ffffff; color: #475569; border-color: #e2e8f0;"
        line_style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-color: #667eea;" if chart_type == "line" else "background: #ffffff; color: #475569; border-color: #e2e8f0;"
        pie_style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-color: #667eea;" if chart_type == "pie" else "background: #ffffff; color: #475569; border-color: #e2e8f0;"
        doughnut_style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-color: #667eea;" if chart_type == "doughnut" else "background: #ffffff; color: #475569; border-color: #e2e8f0;"

        return f"""
<div class="chart-wrapper-card" id="card_{chart_id}" style="
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    width: 100%;
    max-width: 650px;
">
    <h3 style="
        font-size: 16px;
        color: #1e293b;
        margin-bottom: 12px;
        font-family: inherit;
        font-weight: 600;
        text-align: center;
    ">{chart_title}</h3>

    <!-- Dynamic Chart Type Controls -->
    <div class="chart-controls-{chart_id}" style="
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-bottom: 16px;
    ">
        <button class="chart-btn-{chart_id}" onclick="window.changeChartType_{chart_id}('bar', this)" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s; {bar_style}">Bar</button>
        <button class="chart-btn-{chart_id}" onclick="window.changeChartType_{chart_id}('line', this)" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s; {line_style}">Line</button>
        <button class="chart-btn-{chart_id}" onclick="window.changeChartType_{chart_id}('pie', this)" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s; {pie_style}">Pie</button>
        <button class="chart-btn-{chart_id}" onclick="window.changeChartType_{chart_id}('doughnut', this)" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s; {doughnut_style}">Doughnut</button>
    </div>
    
    <div style="position: relative; height: 300px; width: 100%;">
        <canvas id="{chart_id}"></canvas>
    </div>
</div>

<!-- Load Chart.js dynamically inside the rendering template loop as fallback -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
(function() {{
    console.log("[fleetChart] Running chart initialization script for canvas: {chart_id}");
    const ctx = document.getElementById('{chart_id}');
    const card = document.getElementById('card_{chart_id}');
    
    if (!ctx) {{
        console.error("[fleetChart Error] Canvas element with ID '{chart_id}' not found in the DOM.");
        return;
    }}

    function render() {{
        try {{
            console.log("[fleetChart] Initializing Chart.js on canvas: {chart_id}");
            const chartInstance = new Chart(ctx, {{
                type: '{chart_type}',
                data: {{
                    labels: {json_x_labels},
                    datasets: [{{
                        label: '{y_label}',
                        data: {json_y_values},
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.6)',
                            'rgba(118, 75, 162, 0.6)',
                            'rgba(244, 63, 94, 0.6)',
                            'rgba(16, 185, 129, 0.6)',
                            'rgba(245, 158, 11, 0.6)',
                            'rgba(107, 114, 128, 0.6)'
                        ],
                        borderColor: 'rgba(118, 75, 162, 1)',
                        borderWidth: 2,
                        borderRadius: 6,
                        tension: 0.3,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true, position: 'top' }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{ color: '#f1f5f9' }}
                        }},
                        x: {{
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
            
            // Store instance globally for dynamic switching
            window['chartInstance_{chart_id}'] = chartInstance;
            console.log("[fleetChart Success] Chart.js successfully loaded and rendered on canvas: {chart_id}");
        }} catch (renderErr) {{
            console.error("[fleetChart Error] Exception thrown during Chart.js execution:", renderErr);
            showErrorInDOM(renderErr.message);
        }}
    }}

    // Global toggle function bound to this chart_id
    window['changeChartType_{chart_id}'] = function(type, btnElement) {{
        try {{
            const chart = window['chartInstance_{chart_id}'];
            if (!chart) {{
                console.error("[fleetChart Error] Chart instance not found for ID: {chart_id}");
                return;
            }}
            
            console.log("[fleetChart] Toggling type of chart {chart_id} to: " + type);
            
            // Toggle scale displays depending on radial type to prevent layout warnings
            if (type === 'pie' || type === 'doughnut') {{
                if (chart.options.scales && chart.options.scales.x) chart.options.scales.x.display = false;
                if (chart.options.scales && chart.options.scales.y) chart.options.scales.y.display = false;
            }} else {{
                if (chart.options.scales && chart.options.scales.x) chart.options.scales.x.display = true;
                if (chart.options.scales && chart.options.scales.y) chart.options.scales.y.display = true;
            }}
            
            chart.config.type = type;
            chart.update();
            
            // Reset button active styles
            const buttons = document.querySelectorAll('.chart-btn-{chart_id}');
            buttons.forEach(btn => {{
                btn.style.background = '#ffffff';
                btn.style.color = '#475569';
                btn.style.borderColor = '#e2e8f0';
            }});
            
            // Highlight selected button with gradient
            btnElement.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            btnElement.style.color = 'white';
            btnElement.style.borderColor = '#667eea';
            
            console.log("[fleetChart Success] Chart toggled and updated successfully.");
        }} catch (err) {{
            console.error("[fleetChart Error] Failed during type toggle: ", err);
            showErrorInDOM(err.message);
        }}
    }};

    function showErrorInDOM(message) {{
        if (card) {{
            const errorDiv = document.createElement('div');
            errorDiv.style.color = '#ef4444';
            errorDiv.style.fontSize = '12px';
            errorDiv.style.marginTop = '8px';
            errorDiv.style.textAlign = 'center';
            errorDiv.style.fontWeight = 'bold';
            errorDiv.textContent = '⚠️ Error rendering chart: ' + message;
            card.appendChild(errorDiv);
        }}
    }}

    // Check if Chart.js is ready
    if (typeof Chart !== 'undefined') {{
        render();
    }} else {{
        console.warn("[fleetChart Warning] Chart.js library is not yet loaded in scope. Setting up polling check...");
        let checkAttempts = 0;
        const checkInterval = setInterval(() => {{
            checkAttempts++;
            if (typeof Chart !== 'undefined') {{
                clearInterval(checkInterval);
                console.log("[fleetChart] Chart.js loaded successfully after " + (checkAttempts * 50) + "ms. Rendering...");
                render();
            }} else if (checkAttempts > 100) {{  // Timeout after 5 seconds
                clearInterval(checkInterval);
                console.error("[fleetChart Error] Timeout reached waiting for Chart.js CDN script to load.");
                showErrorInDOM("Timeout waiting for Chart.js CDN script to load.");
            }}
        }}, 50);
    }}
}})();
</script>

"""