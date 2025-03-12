import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go  # Graph Objects

def format_float_for_turkish(value):
    """
    Tekil bir float değeri alır, Türkçe formatta (bindelik için nokta,
    ondalık için virgül, iki basamak) string olarak döndürür.
    Örnek: 1234.56 => '1.234,56'
    """
    if pd.isnull(value):
        return ""
    s = f"{value:,.2f}"  # Örn: "1,234.56"
    s = s.replace(",", "X")   # "1X234.56"
    s = s.replace(".", ",")   # "1X234,56"
    s = s.replace("X", ".")   # "1.234,56"
    return s

def format_df_for_turkish(df):
    """
    Bir DataFrame içindeki tüm sayısal sütunları Türkçe formatta stringe çevirir.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].apply(format_float_for_turkish)
    return df_copy

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Menü")
    
    # 1) Excel dosyası yükleme
    uploaded_file = st.sidebar.file_uploader("Excel dosyası yükle", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # 2) Gerekli sayfaları DataFrame olarak oku
            iteration_data = pd.read_excel(uploaded_file, sheet_name="Iteration_Data")
            triangle_data = pd.read_excel(uploaded_file, sheet_name="Triangle")
            totals_data = pd.read_excel(uploaded_file, sheet_name="Totals")
            
            # 3) Triangle sayfasını pivotlayarak kümülatif üçgen elde et
            pivot_triangle = triangle_data.pivot(index="origin", columns="dev", values="value")
            # Dev sütunlarını sıralı hale getirelim (1,2,3,...)
            pivot_triangle = pivot_triangle.reindex(sorted(pivot_triangle.columns), axis=1)
            
            # 4) 6 sekmeli yapı
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Incurred Data", 
                "Iteration Data", 
                "Merging", 
                "Ultimate Graph", 
                "Ultimate Loss per Scenario",
                "Bootstrap Results"
            ])
            
            # ---------------- Tab1: Incurred Data ----------------
            with tab1:
                st.subheader("Incurred Data (Triangle Format)")
                st.write("Bu sekmede, 'Triangle' sayfasından okunan orijinal kümülatif üçgen gösterilmektedir:")
                st.dataframe(format_df_for_turkish(pivot_triangle))
            
            # ---------------- Tab2: Iteration Data ----------------
            with tab2:
                st.subheader("Iteration Data (List Format)")
                st.write("Bu sekmede, 'Iteration_Data' sayfasındaki ham veriyi liste formatında inceleyebilirsiniz:")
                st.dataframe(format_df_for_turkish(iteration_data))
            
            # ---------------- Tab3: Merging ----------------
            with tab3:
                st.subheader("Merging (Orijinal + Tail)")
                st.write("""
                    Bu bölümde, orijinal kümülatif üçgenin son gözlemlenen dev değerinden itibaren, 
                    incremental (iteration) verisinin kümülatif değerlerini ekleyerek tam kümülatif tablo oluşturuyoruz.
                """)
                
                # Kullanıcının seçtiği iteration
                all_iterations = sorted(iteration_data["iteration"].unique())
                selected_iter = st.selectbox("İterasyon Seçiniz", all_iterations)
                
                # Seçili iterasyona ait incremental verileri pivotla
                iter_df = iteration_data[iteration_data["iteration"] == selected_iter].copy()
                pivot_iter = iter_df.pivot(index="origin", columns="dev", values="value")
                pivot_iter = pivot_iter.reindex(sorted(pivot_iter.columns), axis=1)
                
                # cumsum ile incremental => kümülatif
                pivot_iter_cum = pivot_iter.cumsum(axis=1)
                
                st.write("Seçili iterasyonun incremental verilerinin kümülatif hali:")
                st.dataframe(format_df_for_turkish(pivot_iter_cum))
                
                # Orijinal üçgene tail ekleme
                final_cumulative = pivot_triangle.copy()
                # Tüm dev sütunlarını birleştirelim
                all_devs = sorted(list(set(final_cumulative.columns).union(pivot_iter_cum.columns)))
                final_cumulative = final_cumulative.reindex(columns=all_devs)
                
                for origin in final_cumulative.index:
                    row_data = final_cumulative.loc[origin]
                    non_nan_devs = row_data[~row_data.isna()].index
                    if len(non_nan_devs) == 0:
                        # Bu origin için orijinalde veri yoksa pass
                        continue
                    last_obs_dev = max(non_nan_devs)  # Son gözlemlenen dev
                    original_last_cum = final_cumulative.loc[origin, last_obs_dev]
                    
                    # Iteration cumsum'da son gözlemlenen dev değeri
                    iter_last_cum = 0
                    if last_obs_dev in pivot_iter_cum.columns and origin in pivot_iter_cum.index:
                        if pd.notnull(pivot_iter_cum.loc[origin, last_obs_dev]):
                            iter_last_cum = pivot_iter_cum.loc[origin, last_obs_dev]
                    
                    # last_obs_dev'den sonraki dev sütunlarına incremental fark ekle
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
            
            # ---------------- Tab4: Ultimate Graph (Dual Axis) ----------------
           # --- Tab4: Ultimate Graph (Sadece ABS Değerleri) ---
            with tab4:
                st.subheader("Ultimate Değer Grafiği (Sadece ABS)")
                st.write("""
                    Bu grafikte, her bir iterasyonun **mutlak ultimate** (ABS) değerlerini gösteriyoruz. 
                    Sapma (deviation) çizgilerini kaldırdık ve lejandı kapattık.
                """)
                
                # 1) Incremental verilerden her iterasyon için kümülatif değer
                df_group = iteration_data.groupby(["iteration", "dev"])["value"].sum().reset_index()
                df_group = df_group.sort_values(by=["iteration", "dev"])
                df_group["cum_value"] = df_group.groupby("iteration")["value"].cumsum()

                # 2) Her dev için ortalama kümülatif
                avg_df = df_group.groupby("dev")["cum_value"].mean().reset_index().rename(columns={"cum_value": "avg_cum"})
                df_group = df_group.merge(avg_df, on="dev")

                # 3) Deviation hesaplanıyor ama bu sefer sadece ABS'yi kullanacağız
                df_group["deviation"] = df_group["cum_value"] - df_group["avg_cum"]

                # Son devdeki ortalama ultimate
                last_dev = df_group["dev"].max()
                final_avg_ultimate = df_group.loc[df_group["dev"] == last_dev, "avg_cum"].unique()[0]

                # 4) Mutlak değer = final_avg_ultimate + deviation
                df_group["absolute_value"] = final_avg_ultimate + df_group["deviation"]

                # 5) Plotly Graph Objects ile sadece ABS çizgilerini ekle
                fig = go.Figure()
                unique_iters = sorted(df_group["iteration"].unique())

                for it in unique_iters:
                    sub = df_group[df_group["iteration"] == it]
                    fig.add_trace(
                        go.Scatter(
                            x=sub["dev"],
                            y=sub["absolute_value"],
                            mode="lines",
                            name=f"Iter {it}",  # İsterseniz kısaltabilirsiniz
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

            # ---------------- Tab5: Ultimate Loss per Scenario (Histogram) ----------------
           # --- Tab5: Ultimate Loss per Scenario (Histogram) ---
            # --- Tab5: Ultimate Loss per Scenario (Histogram) ---
            with tab5:
                st.subheader("Ultimate Loss per Scenario (Histogram)")
                st.write("""
                    Bu grafikte, Totals sayfasındaki tek sütunluk verideki senaryo değerleri ile 
                    Triangle sayfasından elde edilen (tüm origin'lerin) ultimate incurred toplamını 
                    toplayarak senaryo bazında ultimate loss hesaplıyoruz. 
                    Sonuç değerleri histogram olarak görselleştiriyoruz.
                """)
                
                totals_series = totals_data.iloc[:, 0].dropna()
                ultimate_incurred_series = pivot_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
                ultimate_incurred_total = ultimate_incurred_series.sum()
                
                # Her senaryo için ultimate loss
                ultimate_losses = totals_series + ultimate_incurred_total
                df_ultimate = pd.DataFrame({"Ultimate Loss": ultimate_losses.values})
                
                # Histogram (bin=50, ortalama çizgisi vb.)
                fig2 = px.histogram(
                    df_ultimate,
                    x="Ultimate Loss",
                    nbins=50,
                    title="Senaryo Bazında Ultimate Loss Dağılımı"
                )
                
                # Ortalama çizgisi
                avg_ultimate_loss = df_ultimate["Ultimate Loss"].mean()
                fig2.add_shape(
                    type="line",
                    x0=avg_ultimate_loss,
                    y0=0,
                    x1=avg_ultimate_loss,
                    y1=df_ultimate["Ultimate Loss"].value_counts().max(),
                    line=dict(color="red", width=2, dash="dash"),
                )
                fig2.add_annotation(
                    x=avg_ultimate_loss,
                    y=df_ultimate["Ultimate Loss"].value_counts().max(),
                    text=f"Avg: {format_float_for_turkish(avg_ultimate_loss)}",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(color="red")
                )
                
                # Sütunlar arası boşluk ve kenar çizgisi
                fig2.update_layout(bargap=0.1)
                fig2.update_traces(marker_line_width=1, marker_line_color="white")
                
                # ----------------------------------------------------------------------------------
                # 1) İstediğimiz yüzdelikler: %25, %50, %75, %90, %99.5
                # ----------------------------------------------------------------------------------
                p_list = [25, 50, 75, 90, 99.5]
                tickvals = []
                ticktexts = []
                
                for p in p_list:
                    # Kuantil hesapla
                    q = ultimate_losses.quantile(p / 100.0)
                    # Tick metni: parasal değer + alt satırda % etiketi
                    label = f"{format_float_for_turkish(q)}\n%{p}"
                    tickvals.append(q)
                    ticktexts.append(label)
                
                # ----------------------------------------------------------------------------------
                # 2) X ekseninde sadece bu değerleri göster
                # ----------------------------------------------------------------------------------
                fig2.update_xaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktexts,
                    range=[ultimate_losses.min(), ultimate_losses.max()]  # grafiğin tüm aralığı gösterilsin
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                st.write("""
                    Histogram, ultimate loss değerlerinin dağılımını göstermektedir. 
                    X ekseninde yalnızca seçilen yüzdelik noktalar (%25, %50, %75, %90, %99.5) 
                    gösterilmektedir. Her etiket altında ilgili kuantilin parasal değeri yazılıdır.
                """)

            # ---------------- Tab6: Bootstrap Results (Özet) ----------------
            with tab6:
                st.subheader("Bootstrap Results (Özet)")
                st.write("""
                    Bu sekmede, Totals sayfasından okunan IBNR senaryoları ile 
                    orijinal üçgenin ultimate incurred toplamını kullanarak 
                    çeşitli istatistiksel özetler sunuyoruz.
                """)

                # 1) Totals verisini al (tek sütun) ve NaN'leri çıkar
                totals_series = totals_data.iloc[:, 0].dropna()
                
                # 2) Orijinal üçgenden ultimate_incurred_total hesapla 
                # (her origin'in son diagonel değerinin toplamı)
                ultimate_incurred_series = pivot_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
                ultimate_incurred_total = ultimate_incurred_series.sum()
                
                # 3) Mean IBNR
                mean_ibnr = totals_series.mean()
                # 4) Mean Ultimate (IBNR + orijinal ultimate)
                mean_ultimate = (totals_series + ultimate_incurred_total).mean()
                # 5) Standard Deviation (IBNR)
                std_ibnr = totals_series.std()
                
                # Bu değerleri ekranda gösterelim
                st.write(f"**Mean IBNR:** {format_float_for_turkish(mean_ibnr)}")
                st.write(f"**Mean Ultimate:** {format_float_for_turkish(mean_ultimate)}")
                st.write(f"**Standard Deviation (IBNR):** {format_float_for_turkish(std_ibnr)}")
                
                # 6) VaR / TVaR hesaplamak istediğimiz yüzdelikler
                percentiles = [60, 75, 85, 95, 99.5]
                
                # 7) İlk tablo: Sadece Totals (IBNR) için hesaplanan VaR/TVaR
                rows_ibnr = []
                for p in percentiles:
                    var_ = totals_series.quantile(p / 100.0)
                    tail_values = totals_series[totals_series >= var_]
                    if len(tail_values) > 0:
                        tvar_ = tail_values.mean()
                    else:
                        tvar_ = var_
                    rows_ibnr.append({
                        "Percentile": f"%{p}",
                        "VaR": var_,
                        "TVaR": tvar_
                    })
                
                df_ibnr = pd.DataFrame(rows_ibnr)
                df_ibnr["VaR"] = df_ibnr["VaR"].apply(format_float_for_turkish)
                df_ibnr["TVaR"] = df_ibnr["TVaR"].apply(format_float_for_turkish)
                
                st.write("**VaR / TVaR Değerleri (IBNR - Totals Verisi)**")
                st.table(df_ibnr)
                
                # 8) İkinci tablo: Incurred + IBNR (ultimate loss) için hesaplanan VaR/TVaR
                # Ultimate loss her senaryo için:
                ultimate_losses = totals_series + ultimate_incurred_total
                rows_ultimate = []
                for p in percentiles:
                    var_ult = ultimate_losses.quantile(p / 100.0)
                    tail_vals_ult = ultimate_losses[ultimate_losses >= var_ult]
                    if len(tail_vals_ult) > 0:
                        tvar_ult = tail_vals_ult.mean()
                    else:
                        tvar_ult = var_ult
                    rows_ultimate.append({
                        "Percentile": f"%{p}",
                        "VaR": var_ult,
                        "TVaR": tvar_ult
                    })
                
                df_ultimate_stats = pd.DataFrame(rows_ultimate)
                df_ultimate_stats["VaR"] = df_ultimate_stats["VaR"].apply(format_float_for_turkish)
                df_ultimate_stats["TVaR"] = df_ultimate_stats["TVaR"].apply(format_float_for_turkish)
                
                st.write("**VaR / TVaR Değerleri (Incurred + IBNR - Ultimate Loss)**")
                st.table(df_ultimate_stats)
                
                st.write("""
                    Yukarıdaki tabloda, ilk bölüm Totals verisine (IBNR) ait VaR/TVaR hesaplamalarını, 
                    ikinci bölüm ise orijinal üçgenden elde edilen ultimate incurred toplamı ile Totals verisinin 
                    toplanmasıyla hesaplanan ultimate loss için VaR ve TVaR değerlerini göstermektedir.
                """)

        
        except Exception as e:
            st.error(f"Excel dosyası okunurken bir hata oluştu: {e}")
        finally:
            # 🔥 **Dosya işlemi bitince temizleme adımı**
            if os.path.exists(temp_file_path):  
                os.remove(temp_file_path)  
                st.sidebar.success("Geçici Excel dosyası başarıyla silindi.")

    else:
        st.write("Lütfen sol taraftan bir Excel dosyası yükleyiniz.")

if __name__ == "__main__":
    main()
