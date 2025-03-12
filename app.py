import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import os  # Dosya işlemleri için gerekli

def format_float_for_turkish(value):
    """Sayısal değeri Türkçe formatta gösterir (ondalık virgül, binlik nokta)."""
    if pd.isnull(value):
        return ""
    s = f"{value:,.2f}"  
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  
    return s

def format_df_for_turkish(df):
    """DataFrame içindeki sayısal sütunları Türkçe formatlar."""
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
            # 📌 Geçici dosya olarak kaydet
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2) Excel dosyasını oku
            iteration_data = pd.read_excel(temp_file_path, sheet_name="Iteration_Data")
            triangle_data = pd.read_excel(temp_file_path, sheet_name="Triangle")
            totals_data = pd.read_excel(temp_file_path, sheet_name="Totals")

            # 3) Triangle verisini pivotlayarak üçgen oluştur
            pivot_triangle = triangle_data.pivot(index="origin", columns="dev", values="value")
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
            
            # **Tab1: Incurred Data**
            with tab1:
                st.subheader("Incurred Data (Triangle Format)")
                st.write("Orijinal üçgen verisi:")
                st.dataframe(format_df_for_turkish(pivot_triangle))

            # **Tab2: Iteration Data**
            with tab2:
                st.subheader("Iteration Data (List Format)")
                st.write("Iteration_Data sayfasının ham verisi:")
                st.dataframe(format_df_for_turkish(iteration_data))

            # **Tab3: Merging**
            with tab3:
                st.subheader("Merging (Orijinal + Tail)")
                all_iterations = sorted(iteration_data["iteration"].unique())
                selected_iter = st.selectbox("İterasyon Seçiniz", all_iterations)
                
                iter_df = iteration_data[iteration_data["iteration"] == selected_iter].copy()
                pivot_iter = iter_df.pivot(index="origin", columns="dev", values="value").cumsum(axis=1)

                # Nihai üçgen oluştur
                final_cumulative = pivot_triangle.copy()
                all_devs = sorted(list(set(final_cumulative.columns).union(pivot_iter.columns)))
                final_cumulative = final_cumulative.reindex(columns=all_devs)
                
                for origin in final_cumulative.index:
                    last_obs_dev = max(final_cumulative.loc[origin].dropna().index)
                    original_last_cum = final_cumulative.loc[origin, last_obs_dev]

                    iter_last_cum = 0
                    if last_obs_dev in pivot_iter.columns and origin in pivot_iter.index:
                        iter_last_cum = pivot_iter.loc[origin, last_obs_dev]
                    
                    for dev in all_devs:
                        if dev > last_obs_dev:
                            iter_cum_dev = pivot_iter.loc[origin, dev] if dev in pivot_iter.columns else 0
                            tail_increment = iter_cum_dev - iter_last_cum
                            final_cumulative.loc[origin, dev] = original_last_cum + tail_increment
                
                st.write("Nihai Kümülatif Üçgen:")
                st.dataframe(format_df_for_turkish(final_cumulative))

            # **Tab4: Ultimate Graph**
            with tab4:
                st.subheader("Ultimate Değer Grafiği (Sadece ABS)")
                df_group = iteration_data.groupby(["iteration", "dev"])["value"].sum().reset_index()
                df_group["cum_value"] = df_group.groupby("iteration")["value"].cumsum()
                avg_df = df_group.groupby("dev")["cum_value"].mean().reset_index().rename(columns={"cum_value": "avg_cum"})
                df_group = df_group.merge(avg_df, on="dev")
                df_group["deviation"] = df_group["cum_value"] - df_group["avg_cum"]
                last_dev = df_group["dev"].max()
                final_avg_ultimate = df_group.loc[df_group["dev"] == last_dev, "avg_cum"].unique()[0]
                df_group["absolute_value"] = final_avg_ultimate + df_group["deviation"]

                fig = go.Figure()
                for it in sorted(df_group["iteration"].unique()):
                    sub = df_group[df_group["iteration"] == it]
                    fig.add_trace(go.Scatter(x=sub["dev"], y=sub["absolute_value"], mode="lines", name=f"Iter {it}"))

                fig.update_layout(title="Mutlak Ultimate Değer", xaxis_title="Development Period", yaxis_title="Absolute Ultimate")
                st.plotly_chart(fig, use_container_width=True)

            # **Tab5: Ultimate Loss per Scenario**
            with tab5:
                st.subheader("Ultimate Loss per Scenario (Histogram)")
                totals_series = totals_data.iloc[:, 0].dropna()
                ultimate_losses = totals_series + pivot_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1).sum()
                df_ultimate = pd.DataFrame({"Ultimate Loss": ultimate_losses.values})

                fig2 = px.histogram(df_ultimate, x="Ultimate Loss", nbins=50, title="Ultimate Loss Dağılımı")
                avg_ultimate_loss = df_ultimate["Ultimate Loss"].mean()
                fig2.add_shape(type="line", x0=avg_ultimate_loss, y0=0, x1=avg_ultimate_loss, y1=df_ultimate["Ultimate Loss"].value_counts().max(), line=dict(color="red", width=2, dash="dash"))
                st.plotly_chart(fig2, use_container_width=True)

            # **Tab6: Bootstrap Results**
            with tab6:
                st.subheader("Bootstrap Results (Özet)")
                mean_ibnr = totals_series.mean()
                mean_ultimate = (totals_series + pivot_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1).sum()).mean()
                st.write(f"**Mean IBNR:** {format_float_for_turkish(mean_ibnr)}")
                st.write(f"**Mean Ultimate:** {format_float_for_turkish(mean_ultimate)}")

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
