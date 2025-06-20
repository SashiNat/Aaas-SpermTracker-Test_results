import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
from io import BytesIO
import base64
from pathlib import Path
import time

# Handle Plotly import with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Plotly not installed - charts will be disabled")
    PLOTLY_AVAILABLE = False

# Handle OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {str(e)}")
    st.error("Please install OpenCV properly. See troubleshooting guide below.")
    CV2_AVAILABLE = False

# Only import analyzer if OpenCV is available
if CV2_AVAILABLE:
    try:
        from SpermTracker import SpermMotilityAnalyzer
        ANALYZER_AVAILABLE = True
    except ImportError as e:
        st.error(f"❌ Sperm analyzer import failed: {str(e)}")
        st.info("Make sure 'SpermTracker.py' is in the same directory as this Streamlit app")
        ANALYZER_AVAILABLE = False
else:
    ANALYZER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sperm Motility Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

def get_download_link(file_path, link_text, file_name):
    """Generate download link for files"""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" style="text-decoration: none; color: white; background-color: #1f77b4; padding: 8px 16px; border-radius: 4px; display: inline-block; margin: 4px;">{link_text}</a>'
        return href
    return ""

def create_download_zip(output_dir):
    """Create a zip file with all outputs"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def display_video_info(video_path):
    """Display video information"""
    if not CV2_AVAILABLE:
        st.error("OpenCV not available - cannot display video info")
        return
        
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Resolution", f"{width}×{height}")
    with col2:
        st.metric("Frame Rate", f"{fps} FPS")
    with col3:
        st.metric("Duration", f"{duration:.1f}s")
    with col4:
        st.metric("Total Frames", f"{frame_count:,}")

def create_results_dashboard(df):
    """Create interactive dashboard for results"""
    st.subheader("📊 Analysis Dashboard")
    
    # Summary metrics
    total_fast = df['Fast'].sum()
    total_slow = df['Slow'].sum()
    total_immotile = df['Immotile'].sum()
    total_detections = total_fast + total_slow + total_immotile
    
    if total_detections > 0:
        motile_percentage = (total_fast + total_slow) / total_detections * 100
        
        # Classification
        if motile_percentage >= 40:
            classification = "Normal Motility"
            color = "green"
        elif motile_percentage >= 32:
            classification = "Below Normal"
            color = "orange"
        else:
            classification = "Poor Motility"
            color = "red"
        
        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Detections", f"{total_detections:,}")
        with col2:
            st.metric("Fast Moving", f"{total_fast:,}", f"{total_fast/total_detections*100:.1f}%")
        with col3:
            st.metric("Slow Moving", f"{total_slow:,}", f"{total_slow/total_detections*100:.1f}%")
        with col4:
            st.metric("Immotile", f"{total_immotile:,}", f"{total_immotile/total_detections*100:.1f}%")
        with col5:
            st.metric("Overall Motility", f"{motile_percentage:.1f}%", classification)
        
        # Classification alert
        st.markdown(f"""
        <div class="{'success-box' if color == 'green' else 'info-box'}">
            <strong>Motility Classification:</strong> {classification} ({motile_percentage:.1f}% motile)
        </div>
        """, unsafe_allow_html=True)
        
        # Create interactive plots if Plotly is available
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of movement distribution
                fig_pie = px.pie(
                    values=[total_fast, total_slow, total_immotile],
                    names=['Fast Moving', 'Slow Moving', 'Immotile'],
                    title="Movement Distribution",
                    color_discrete_map={
                        'Fast Moving': '#00ff00',
                        'Slow Moving': '#ffff00', 
                        'Immotile': '#ff0000'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Time series plot
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=df['Frame #'], y=df['Fast'], name='Fast', line=dict(color='green')))
                fig_time.add_trace(go.Scatter(x=df['Frame #'], y=df['Slow'], name='Slow', line=dict(color='orange')))
                fig_time.add_trace(go.Scatter(x=df['Frame #'], y=df['Immotile'], name='Immotile', line=dict(color='red')))
                fig_time.update_layout(title="Sperm Count Over Time", xaxis_title="Frame Number", yaxis_title="Count")
                st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("📊 Install Plotly to see interactive charts: `pip install plotly`")

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Sperm Motility Analysis System</h1>', unsafe_allow_html=True)
    
    # Check if system is ready
    if not CV2_AVAILABLE or not ANALYZER_AVAILABLE:
        st.error("❌ System not ready - missing dependencies")
        
        st.markdown("""
        ## 🔧 Troubleshooting Installation
        
        ### **OpenCV Installation:**
        ```bash
        conda activate sperm
        pip uninstall opencv-python
        conda install -c conda-forge opencv
        ```
        
        ### **Alternative OpenCV Installation:**
        ```bash
        pip uninstall opencv-python opencv-contrib-python
        pip install opencv-python-headless==4.8.0.74
        ```
        
        ### **Plotly Installation (Optional):**
        ```bash
        pip install plotly
        ```
        
        **After installation, restart your terminal and run:** `streamlit run streamlit_app.py`
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Analysis Settings")
        
        # Upload section
        st.subheader("📁 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a sperm motility video",
            type=['mp4', 'avi', 'mov'],
            help="Upload an MP4 video file of sperm motility for analysis"
        )
        
        # Analysis options
        st.subheader("⚙️ Options")
        debug_mode = st.checkbox("Enable Debug Mode", help="Save debug images for the first 3 frames")
        show_progress = st.checkbox("Show Detailed Progress", value=True)
        
        # Info section
        st.subheader("ℹ️ About")
        st.info("""
        This system analyzes sperm motility using computer vision:
        
        **Detection Patterns:**
        - 🔵 Circular heads
        - 🥚 Oval/elongated 
        - 👥 Double-headed
        - ⭐ Halo-effect
        
        **Classification:**
        - 🟢 Fast moving
        - 🟡 Slow moving  
        - 🔴 Immotile
        
        **Outputs:**
        - 📄 CSV data report
        - 🎥 Trajectory videos (2 types)
        - 📊 Motility analysis
        """)
    
    # Main content area
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # Display video information
        st.subheader("📹 Video Information")
        display_video_info(video_path)
        
        # Video preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎥 Video Preview")
            st.video(uploaded_file)
        
        with col2:
            st.subheader("🚀 Analysis Control")
            
            # Analysis button
            if st.button("🔬 Start Analysis", type="primary", use_container_width=True):
                # Create output directory
                output_dir = tempfile.mkdtemp()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize analyzer
                    status_text.text("🔧 Initializing analyzer...")
                    progress_bar.progress(10)
                    analyzer = SpermMotilityAnalyzer()
                    
                    # Start analysis
                    status_text.text("🔍 Starting video analysis...")
                    progress_bar.progress(20)
                    
                    # Run analysis with progress updates
                    start_time = time.time()
                    results_df = analyzer.analyze_video(
                        video_path, 
                        output_dir=output_dir,
                        debug_mode=debug_mode
                    )
                    
                    analysis_time = time.time() - start_time
                    progress_bar.progress(100)
                    status_text.text(f"✅ Analysis completed in {analysis_time:.1f}s")
                    
                    # Store results in session state
                    st.session_state['results_df'] = results_df
                    st.session_state['output_dir'] = output_dir
                    st.session_state['analysis_completed'] = True
                    
                    st.success("🎉 Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("❌ Analysis failed")
                
                finally:
                    # Cleanup temporary video file
                    if os.path.exists(video_path):
                        os.unlink(video_path)
        
        # Display results if analysis completed
        if st.session_state.get('analysis_completed', False):
            st.markdown("---")
            
            # Results dashboard
            results_df = st.session_state['results_df']
            output_dir = st.session_state['output_dir']
            
            create_results_dashboard(results_df)
            
            # Data table
            st.subheader("📋 Detailed Results")
            
            # Add summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Average counts per frame:**")
                avg_stats = {
                    'Fast': results_df['Fast'].mean(),
                    'Slow': results_df['Slow'].mean(),
                    'Immotile': results_df['Immotile'].mean()
                }
                st.write(avg_stats)
            
            with col2:
                st.write("**Total counts:**")
                total_stats = {
                    'Fast': results_df['Fast'].sum(),
                    'Slow': results_df['Slow'].sum(), 
                    'Immotile': results_df['Immotile'].sum()
                }
                st.write(total_stats)
            
            # Show data table with pagination
            st.write("**Frame-by-frame data:**")
            page_size = 50
            total_rows = len(results_df)
            total_pages = (total_rows - 1) // page_size + 1
            
            if total_pages > 1:
                page = st.selectbox("Select page:", range(1, total_pages + 1))
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_rows)
                display_df = results_df.iloc[start_idx:end_idx]
                st.write(f"Showing rows {start_idx + 1}-{end_idx} of {total_rows}")
            else:
                display_df = results_df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # CSV download
                csv_path = os.path.join(output_dir, 'sperm_motility_analysis.csv')
                if os.path.exists(csv_path):
                    with open(csv_path, 'rb') as f:
                        csv_data = f.read()
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name="sperm_analysis.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Dots video download
                dots_path = os.path.join(output_dir, 'sperm_analysis_dots_trajectories.mp4')
                if os.path.exists(dots_path):
                    with open(dots_path, 'rb') as f:
                        video_data = f.read()
                    st.download_button(
                        label="🎥 Download Dots Video",
                        data=video_data,
                        file_name="dots_trajectories.mp4",
                        mime="video/mp4"
                    )
            
            with col3:
                # Bbox video download
                bbox_path = os.path.join(output_dir, 'sperm_analysis_bbox_trajectories.mp4')
                if os.path.exists(bbox_path):
                    with open(bbox_path, 'rb') as f:
                        video_data = f.read()
                    st.download_button(
                        label="📦 Download Bbox Video",
                        data=video_data,
                        file_name="bbox_trajectories.mp4",
                        mime="video/mp4"
                    )
            
            with col4:
                # All files zip download
                try:
                    zip_data = create_download_zip(output_dir)
                    st.download_button(
                        label="📁 Download All Files",
                        data=zip_data,
                        file_name="sperm_analysis_complete.zip",
                        mime="application/zip"
                    )
                except Exception as e:
                    st.error(f"Error creating zip: {str(e)}")
            
            # Debug files download
            if debug_mode:
                st.subheader("🔍 Debug Files")
                debug_files = [f for f in os.listdir(output_dir) if f.startswith('debug_')]
                if debug_files:
                    st.write("Debug images generated for first 3 frames:")
                    for debug_file in sorted(debug_files):
                        if debug_file.endswith('.jpg'):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📸 {debug_file}")
                            with col2:
                                debug_path = os.path.join(output_dir, debug_file)
                                with open(debug_path, 'rb') as f:
                                    debug_data = f.read()
                                st.download_button(
                                    label="Download",
                                    data=debug_data,
                                    file_name=debug_file,
                                    mime="image/jpeg",
                                    key=debug_file
                                )
    
    else:
        # Welcome screen - Simple version
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Upload your sperm motility video** (MP4 format recommended)
        2. **Configure analysis settings** in the sidebar (optional)
        3. **Click "Start Analysis"** to begin processing
        4. **Download results** and review the analysis dashboard
        
        ## 📊 Expected Output Results
        
        **CSV Report:**
        - Frame-by-frame sperm counts (Frame# | Fast | Slow | Immotile)
        
        **Trajectory Videos:**
        - Dots with trajectories video (colored circles showing movement paths)
        - Bounding boxes video (detection rectangles with movement trails)
        
        **Analysis Dashboard:**
        - Movement distribution charts
        - Motility classification  
        - Interactive data visualization
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    
    main()