import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def create_risk_chart_image(prob, sleep, lifestyle):
    """Generate chart as matplotlib figure and return as BytesIO image."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    
    metrics = ["Risk Probability", "Sleep Health", "Activity Level"]
    values = [prob * 10, sleep, 8 if lifestyle == "Active" else (5 if lifestyle == "Sedentary" else 2)]
    colors_list = ['#cc0000', '#4CAF50', '#2196F3']
    
    bars = ax.bar(metrics, values, color=colors_list, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score (0-10)', fontsize=12, fontweight='bold')
    ax.set_title('Risk Assessment Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    
    return img_buffer


def generate_pdf(user_input, age, gender, height, weight, sleep, lifestyle, 
                 prob, risk_level, recommendation):
    """
    Generate a structured PDF report with all analysis details.
    
    Returns: BytesIO object containing the PDF
    """
    
    # Create BytesIO buffer for PDF
    pdf_buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Container for PDF elements
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#000000'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#cc0000'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14
    )
    
    info_style = ParagraphStyle(
        'Info',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        leading=12
    )
    
    # --- HEADER ---
    elements.append(Paragraph("🧠 MindGuard: Suicide Ideation Detection Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", info_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # --- USER INPUT SECTION ---
    elements.append(Paragraph("📝 User Input", heading_style))
    user_input_text = user_input if user_input.strip() else "[No input provided]"
    # Truncate very long inputs
    if len(user_input_text) > 500:
        user_input_text = user_input_text[:500] + "...[truncated]"
    elements.append(Paragraph(f"<i>{user_input_text}</i>", info_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # --- DEMOGRAPHIC INFORMATION ---
    elements.append(Paragraph("👤 Demographic Information", heading_style))
    
    demo_data = [
        ['Age', f'{age} years'],
        ['Gender', gender],
        ['Height', f'{height} cm'],
        ['Weight', f'{weight} kg'],
        ['Sleep Quality', f'{sleep} hours/night'],
        ['Lifestyle', lifestyle]
    ]
    
    demo_table = Table(demo_data, colWidths=[2*inch, 2.5*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    elements.append(demo_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # --- RISK ASSESSMENT RESULTS ---
    elements.append(Paragraph("⚠️ Risk Assessment Results", heading_style))
    
    # Risk level with color coding
    risk_color_map = {
        'Critical': '#cc0000',
        'High': '#ff6600',
        'Medium': '#ffcc00',
        'Low': '#00cc00'
    }
    risk_color = risk_color_map.get(risk_level, '#000000')
    
    result_data = [
        ['Risk Level', f'{risk_level}'],
        ['Confidence Score', f'{prob*100:.1f}%'],
        ['Recommendation', recommendation]
    ]
    
    result_table = Table(result_data, colWidths=[1.5*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # --- CHART ---
    elements.append(Paragraph("📊 Metrics Visualization", heading_style))
    try:
        chart_image = create_risk_chart_image(prob, sleep, lifestyle)
        img = Image(chart_image, width=5*inch, height=3*inch)
        elements.append(img)
    except Exception as e:
        elements.append(Paragraph(f"[Chart generation failed: {str(e)}]", info_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # --- METHODOLOGY ---
    elements.append(Paragraph("🔬 Methodology", heading_style))
    methodology_text = """
    <b>Preprocessing:</b> NLTK Tokenization and Lemmatization<br/>
    <b>Features:</b> Hybrid Unigram & Bigram TF-IDF (25,000 features)<br/>
    <b>Ensemble:</b> Soft Voting (Logistic Regression + Random Forest)<br/>
    <b>Threshold:</b> 0.4 Sensitivity Adjustment for Clinical Safety<br/>
    <b>Risk Classification:</b><br/>
    • Low: Probability &lt; 0.4<br/>
    • Medium: 0.4 ≤ Probability &lt; 0.6<br/>
    • High: 0.6 ≤ Probability &lt; 0.8<br/>
    • Critical: Probability ≥ 0.8
    """
    elements.append(Paragraph(methodology_text, normal_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    # --- DISCLAIMER ---
    disclaimer_text = """
    <b>⚠️ DISCLAIMER:</b> This tool is for informational purposes only and should not be used as a substitute 
    for professional mental health evaluation or treatment. If you or someone you know is in crisis, please contact 
    emergency services or a mental health professional immediately.
    """
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#cc0000'),
        leading=11,
        alignment=TA_CENTER,
        borderColor=colors.HexColor('#cc0000'),
        borderWidth=1,
        borderPadding=8,
        backColor=colors.HexColor('#ffe6e6')
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    return pdf_buffer
