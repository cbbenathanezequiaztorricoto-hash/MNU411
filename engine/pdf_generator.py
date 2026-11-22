import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class PDFReportEngine:

    @staticmethod
    def generar_reporte_pdf(
        titulo_metodo: str,
        entradas_dict: dict,
        resultado_str: str,
        iteraciones_list: list = None,
        pasos_matriz_list: list = None,
    ) -> bytes:
        """
        Genera un reporte académico en PDF detallando el procedimiento
        paso a paso (tablas de iteración o reducción matricial).
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )
        story = []
        styles = getSampleStyleSheet()

        # Estilos de encabezado
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=15,
            textColor=colors.HexColor("#1E3A8A"),
            spaceAfter=8,
        )
        h2_style = ParagraphStyle(
            "Heading2",
            parent=styles["Heading2"],
            fontSize=11,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=8,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyText", parent=styles["Normal"], fontSize=9, spaceAfter=4
        )

        story.append(Paragraph(f"MNU411 - Reporte de Procedimiento: {titulo_metodo}", title_style))
        story.append(Spacer(1, 4))

        # 1. Parámetros de Entrada
        story.append(Paragraph("1. Parámetros de Entrada", h2_style))
        input_data = [["Parámetro", "Valor Evaluado"]]
        for k, v in entradas_dict.items():
            input_data.append([str(k), str(v)])

        t_inputs = Table(input_data, colWidths=[180, 320])
        t_inputs.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(t_inputs)
        story.append(Spacer(1, 8))

        # 2. Resultado Obtenido
        story.append(Paragraph("2. Solución Declarada", h2_style))
        story.append(Paragraph(f"<b>Resultado Final:</b> {resultado_str}", body_style))
        story.append(Spacer(1, 8))

        # 3. Procedimiento Paso a Paso: Reducción Matricial (Gauss / Gauss-Jordan)
        if pasos_matriz_list and len(pasos_matriz_list) > 0:
            story.append(Paragraph("3. Procedimiento de Reducción Matricial Paso a Paso", h2_style))

            for pm in pasos_matriz_list:
                desc = f"<b>{pm['paso']}</b> — Fórmula de Renglón: <i>{pm['formula']}</i>"
                story.append(Paragraph(desc, body_style))

                # Renderizar la matriz del paso
                mat_data = [[str(val) for val in row] for row in pm["matriz"]]
                t_mat = Table(mat_data)
                t_mat.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9CA3AF")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ])
                )
                story.append(t_mat)
                story.append(Spacer(1, 6))

        # 4. Procedimiento Paso a Paso: Tabla de Iteraciones (Newton, Bisección, Jacobi, etc.)
        elif iteraciones_list and len(iteraciones_list) > 0:
            story.append(Paragraph("3. Tabla de Iteraciones (i | Xn | Error)", h2_style))
            headers = list(iteraciones_list[0].keys())
            table_data = [headers]

            for row in iteraciones_list:
                table_data.append([str(row[h]) for h in headers])

            t_iters = Table(table_data)
            t_iters.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9CA3AF")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ])
            )
            story.append(t_iters)

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes