import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import os

def format_float_for_turkish(value):
    if pd.isnull(value):
        return ""
    s = f"{value:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def format_df_for_turkish(df):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].apply(format_float_for_turkish)
    return df_copy

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Menü")
    
    # Excel dosyası yükleme
    uploaded_file = st.sidebar.file_uploader("Excel dosyası yükle", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Geçici dosya yolu belirle
        temp_file_path = f"temp_{uploaded_file.name}"
        try:
            # Dosyayı geçici olarak kaydet
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Excel dosyasını oku
            iteration_data = pd.read_excel(temp_file_path, sheet_name="Iteration_Data")
            triangle_data = pd.read_excel(temp_file_path, sheet_name="Triangle")
            totals_data = pd.read_excel(temp_file_path, sheet_name="Totals")
            
            # Triangle verisini pivotlayarak kümülatif üçgen elde et
            pivot_triangle = triangle_data.pivot(index="origin", columns="dev", values="value")
            pivot_triangle = pivot_triangle.reindex(sorted(pivot_triangle.columns), axis=1)
            
            # 6 sekmeli yapı
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Incurred Data", 
                "Iteration Data", 
                "Merging", 
                "Ultimate Graph", 
                "Ultimate Loss per Scenario",
                "Bootstrap Results"
            ])
            
            with tab1:
                st.subheader("Incurred Data (Triangle Format)")
                st.write("Orijinal kümülatif üçgen:")
                st.dataframe(format_df_for_turkish(pivot_triangle))
            
            with tab2:
                st.subheader("Iteration Data (List Format)")
                st.write("Iteration_Data sayfasının ham verisi:")
                st.dataframe(format_df_for_turkish(iteration_data))
            
            with tab3:
                st.subheader("Merging (Orijinal + Tail)")
                all_iterations = sorted(iteration_data["iteration"].unique())
                selected_iter = st.selectbox("İterasyon Seçiniz", all_iterations)
                
                iter_df = iteration_data[iteration_data["iteration"] == selected_iter].copy()
                pivot_iter = iter_df.pivot(index="origin", columns="dev", values="value")
                pivot_iter = pivot_iter.reindex(sorted(pivot_iter.columns), axis=1)
                pivot_iter_cum = pivot_iter.cumsum(axis=1)
                
                st.write("Seçili iterasyonun incremental verilerinin kümülatif hali:")
                st.dataframe(format_df_for_turkish(pivot_iter_cum))
                
                final_cumulative = pivot_triangle.copy()
                all_devs = sorted(list(set(final_cumulative.columns).union(pivot_iter_cum.columns)))
                final_cumulative = final_cumulative.reindex(columns=all_devs)
                
                for origin in final_cumulative.index:
                    row_data = final_cumulative.loc[origin]
                    non_nan_devs = row_data[~row_data.isna()].index
                    if len(non_nan_devs) == 0:
                        continue
                    last_obs_dev = max(non_nan_devs)
                    original_last_cum = final_cumulative.loc[origin, last_obs_dev]
                    
                    iter_last_cum = 0
                    if last_obs_dev in pivot_iter_cum.columns and origin in pivot_iter_cum.index:
                        if pd.notnull(pivot_iter_cum.loc[origin, last_obs_dev]):
                            iter_last_cum = pivot_iter_cum.loc[origin, last_obs_dev]
                    
                    for dev in all_devs:
                        if dev > last_obs_dev:
                            iter_cum_dev = 0
                            if origin in pivot_iter_cum.index and dev in pivot_iter_cum.columns:
                                val = pivot_iter_cum.loc[origin, dev]
                                if pd.notnull(val):
                                    iter_cum_dev = val
                            tail_increment = iter_cum_dev - iter_last_cum
                            final_cumulative.loc[origin, dev] = original_last_cum + tail_increment
                
                st.write("Nihai Kümülatif Üçgen (Orijinal + Tail):")
                st.dataframe(format_df_for_turkish(final_cumulative))
            
               with tab4:
                    st.subheader("Ultimate Değer Grafiği (Sadece ABS)")
                    # 1) Incremental verilerden her iterasyon için kümülatif değer
                    df_group = iteration_data.groupby(["iteration", "dev"])["value"].sum().reset_index()
                    df_group = df_group.sort_values(by=["iteration", "dev"])
                    df_group["cum_value"] = df_group.groupby("iteration")["value"].cumsum()
                
                    # 2) Her dev için ortalama kümülatif değeri hesapla
                    avg_df = df_group.groupby("dev")["cum_value"].mean().reset_index().rename(columns={"cum_value": "avg_cum"})
                    df_group = df_group.merge(avg_df, on="dev")
                
                    # 3) Deviation hesaplanıyor: kümülatif değerden ortalamayı çıkar
                    df_group["deviation"] = df_group["cum_value"] - df_group["avg_cum"]
                
                    # 4) Son devdeki ortalama ultimate değeri
                    last_dev = df_group["dev"].max()
                    final_avg_ultimate = df_group.loc[df_group["dev"] == last_dev, "avg_cum"].unique()[0]
                
                    # 5) Mutlak değer hesapla: final ortalama + deviation
                    df_group["absolute_value"] = final_avg_ultimate + df_group["deviation"]
                
                    # Grafik oluşturma
                    fig = go.Figure()
                    for it in sorted(df_group["iteration"].unique()):
                        sub = df_group[df_group["iteration"] == it]
                        fig.add_trace(
                            go.Scatter(
                                x=sub["dev"],
                                y=sub["absolute_value"],
                                mode="lines",
                                name=f"Iter {it}"
                            )
                        )
                
                    # 6) Grafik düzenlemeleri
                    fig.update_layout(
                        title="Mutlak Ultimate Değer (Sadece ABS)",
                        xaxis=dict(title="Development Period"),
                        yaxis=dict(title="Absolute Ultimate"),
                        showlegend=False,  # Lejandı tamamen kapatıyoruz
                        hovermode="x unified"
                    )
                    fig.update_layout(
                        yaxis=dict(
                            side="right",       # Eksen sağda
                            anchor="free",      # Serbest konum
                            overlaying="free",  # Başka eksene bindirme yok
                            position=1.0        # X ekseni üzerinde 1.0 konumunda (sağ kenar)
                        )
                    )
                    fig.update_layout(
                        hovermode="closest"
                    )
                
                    # Eğer bir yatay çizgi veya annotation eklemek istemiyorsanız, eklemeyin.
                    # (Sapma=0 çizgisi gibi unsurlar devre dışı bırakıldı.)
                
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("""
                        Her çizgi, ilgili iterasyonun development period boyunca **mutlak ultimate** 
                        (final ortalama + sapma) değerlerini göstermektedir. Lejand ve sapma çizgileri kaldırılmıştır.
                    """)

            with tab5:
                st.subheader("Ultimate Loss per Scenario (Histogram)")
                totals_series = totals_data.iloc[:, 0].dropna()
                ultimate_incurred_series = pivot_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
                ultimate_incurred_total = ultimate_incurred_series.sum()
                ultimate_losses = totals_series + ultimate_incurred_total
                df_ultimate = pd.DataFrame({"Ultimate Loss": ultimate_losses.values})
                
                fig2 = px.histogram(df_ultimate, x="Ultimate Loss", nbins=50, title="Senaryo Bazında Ultimate Loss Dağılımı")
                avg_ultimate_loss = df_ultimate["Ultimate Loss"].mean()
                fig2.add_shape(
                    type="line",
                    x0=avg_ultimate_loss,
                    y0=0,
                    x1=avg_ultimate_loss,
                    y1=df_ultimate["Ultimate Loss"].value_counts().max(),
                    line=dict(color="red", width=2, dash="dash")
                )
                fig2.add_annotation(
                    x=avg_ultimate_loss,
                    y=df_ultimate["Ultimate Loss"].value_counts().max(),
                    text=f"Avg: {format_float_for_turkish(avg_ultimate_loss)}",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(color="red")
                )
                fig2.update_layout(bargap=0.1)
                fig2.update_traces(marker_line_width=1, marker_line_color="white")
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab6:
                st.subheader("Bootstrap Results (Özet)")
                totals_series = totals_data.iloc[:, 0].dropna()
                ultimate_incurred_series = pivot_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
                ultimate_incurred_total = ultimate_incurred_series.sum()
                mean_ibnr = totals_series.mean()
                mean_ultimate = (totals_series + ultimate_incurred_total).mean()
                std_ibnr = totals_series.std()
                
                st.write(f"**Mean IBNR:** {format_float_for_turkish(mean_ibnr)}")
                st.write(f"**Mean Ultimate:** {format_float_for_turkish(mean_ultimate)}")
                st.write(f"**Standard Deviation (IBNR):** {format_float_for_turkish(std_ibnr)}")
                
                percentiles = [60, 75, 85, 95, 99.5]
                rows_ibnr = []
                for p in percentiles:
                    var_ = totals_series.quantile(p / 100.0)
                    tail_values = totals_series[totals_series >= var_]
                    tvar_ = tail_values.mean() if len(tail_values) > 0 else var_
                    rows_ibnr.append({"Percentile": f"%{p}", "VaR": var_, "TVaR": tvar_})
                
                df_ibnr = pd.DataFrame(rows_ibnr)
                df_ibnr["VaR"] = df_ibnr["VaR"].apply(format_float_for_turkish)
                df_ibnr["TVaR"] = df_ibnr["TVaR"].apply(format_float_for_turkish)
                st.write("**VaR / TVaR Değerleri (IBNR - Totals Verisi)**")
                st.table(df_ibnr)
                
                ultimate_losses = totals_series + ultimate_incurred_total
                rows_ultimate = []
                for p in percentiles:
                    var_ult = ultimate_losses.quantile(p / 100.0)
                    tail_vals_ult = ultimate_losses[ultimate_losses >= var_ult]
                    tvar_ult = tail_vals_ult.mean() if len(tail_vals_ult) > 0 else var_ult
                    rows_ultimate.append({"Percentile": f"%{p}", "VaR": var_ult, "TVaR": tvar_ult})
                
                df_ultimate_stats = pd.DataFrame(rows_ultimate)
                df_ultimate_stats["VaR"] = df_ultimate_stats["VaR"].apply(format_float_for_turkish)
                df_ultimate_stats["TVaR"] = df_ultimate_stats["TVaR"].apply(format_float_for_turkish)
                st.write("**VaR / TVaR Değerleri (Incurred + IBNR - Ultimate Loss)**")
                st.table(df_ultimate_stats)
                st.write("Yukarıdaki tabloda ilgili VaR ve TVaR hesaplamaları gösterilmiştir.")
        
        except Exception as e:
            st.error(f"Excel dosyası okunurken bir hata oluştu: {e}")
        finally:
            # Dosya var mı kontrol edip sil
            if os.path.exists(temp_file_path):  
                os.remove(temp_file_path)
                st.sidebar.success("Geçici Excel dosyası başarıyla silindi.")
    
    else:
        st.write("Lütfen sol taraftan bir Excel dosyası yükleyiniz.")

if __name__ == "__main__":
    main()
